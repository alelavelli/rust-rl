use rand::Rng;
use rand_distr::Distribution;
use rlenv::tabular::TabularEnvironment;

use crate::{learn::LearningError, policy::tabular::TabularPolicy};

/// Sarsa: on-policy TD, update rule:
/// $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right] $$
///
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `episodes`: number of episodes to generate
/// - `episode_max_len`: maximum length of an episode
/// - `gamma`: discount factor
/// - `step_size`: step size for q update
/// - `expected`: true to use expected sarsa instead of standard sarsa learning algorithm
/// - `render_env`: if true render the environment during the learning
/// - `rng`: random generator
///
///
/// ## Expected Sarsa
///
/// Expected Sarsa is a off-policy learning algorithm similar to Q-learning that instead
/// of taking the maximum over next state-acation pair it uses the expected value, taking
/// into account how likely each action is under the current policy
/// $$ Q(S, A) \leftarrow Q(S_t, A_t) + \alpha \left[ R + \gamma \sum_a \pi(a | S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right] $$
///
///
#[allow(clippy::too_many_arguments)]
pub fn sarsa<P, E, R>(
    policy: &mut P,
    environment: &mut E,
    episodes: i32,
    episode_max_len: i32,
    gamma: f32,
    step_size: f32,
    expected: bool,
    render_env: bool,
    rng: &mut R,
) -> Result<(), LearningError>
where
    P: TabularPolicy,
    E: TabularEnvironment,
    R: Rng + ?Sized,
{
    // Q function initialization
    // Q table can have arbitrarily values except Q(terminal, •) which needs to be 0
    // therefore we set the value for each terminal state to 0
    for terminal_state in environment.get_terminal_states() {
        for i in 0..environment.get_number_actions() {
            policy.update_q_entry(terminal_state, i, 0.0);
        }
    }
    // loop for each episode
    for _ in 0..episodes {
        // init environment
        let mut state = environment.reset();
        // choose action from S with policy
        let mut action = policy.step(state, rng).map_err(LearningError::PolicyStep)?;

        let mut step_number = 0;
        // loop until is S is terminal
        loop {
            step_number += 1;
            // take action A and observer R and S'
            let episode_step = environment
                .step(action, rng)
                .map_err(LearningError::EnvironmentStep)?;

            let q_sa = policy.get_q_value(state, action);

            if expected {
                let q_expected = policy.expected_q_value(episode_step.state);

                let new_q_value =
                    q_sa + step_size * (episode_step.reward + gamma * q_expected - q_sa);
                // update q entry
                policy.update_q_entry(state, action, new_q_value);

                action = policy
                    .step(episode_step.state, rng)
                    .map_err(LearningError::PolicyStep)?;
            } else {
                // choose A' from S' with policy
                let a_prime = policy
                    .step(episode_step.state, rng)
                    .map_err(LearningError::PolicyStep)?;

                // update q entry with Q(S, A) = Q(S, A) + step_size [ R + gamma * Q(S', A') - Q(S, A) ]
                let q_spap = policy.get_q_value(episode_step.state, a_prime);
                let new_q_value = q_sa + step_size * (episode_step.reward + gamma * q_spap - q_sa);
                // update q entry
                policy.update_q_entry(state, action, new_q_value);
                // set A = A'
                action = a_prime;
            }

            // set S = S'
            state = episode_step.state;

            if render_env {
                environment.render();
            }

            if episode_step.terminated | (step_number >= episode_max_len) {
                break;
            }
        }
    }
    Ok(())
}

/// Q-Learning, update rule
/// $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right] $$
///
/// Off-policy TD control algorithm
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `episodes`: number of episodes to generate
/// - `episode_max_len`: maximum length of an episode
/// - `gamma`: discount factor
/// - `step_size`: step size for q update
///
/// ## Returns
///
/// `new_policy`: new policy that was optimized over the environment
#[allow(clippy::too_many_arguments)]
pub fn qlearning<P, E, R>(
    policy: &mut P,
    environment: &mut E,
    episodes: i32,
    episode_max_len: i32,
    gamma: f32,
    step_size: f32,
    render_env: bool,
    rng: &mut R,
) -> Result<(), LearningError>
where
    P: TabularPolicy,
    E: TabularEnvironment,
    R: Rng + ?Sized,
{
    // Q function initialization
    // Q table can have arbitrarily values except Q(terminal, •) which needs to be 0
    // therefore we set the value for each terminal state to 0
    for terminal_state in environment.get_terminal_states() {
        for i in 0..environment.get_number_actions() {
            policy.update_q_entry(terminal_state, i, 0.0);
        }
    }
    // loop for each episode
    for _ in 0..episodes {
        // init environment
        let mut state = environment.reset();

        let mut step_number = 0;
        // loop until is S is terminal
        loop {
            // choose action from S with policy
            let action = policy.step(state, rng).map_err(LearningError::PolicyStep)?;

            step_number += 1;
            // take action A and observer R and S'
            let episode_step = environment
                .step(action, rng)
                .map_err(LearningError::EnvironmentStep)?;

            // update q entry with Q(S, A) = Q(S, A) + step_size [ R + gamma * max_a Q(S', a) - Q(S, A) ]
            let q_sa = policy.get_q_value(state, action);
            let q_max = policy
                .get_max_q_value(episode_step.state)
                .map_err(LearningError::PolicyStep)?;
            let new_q_value = q_sa + step_size * (episode_step.reward + gamma * q_max - q_sa);
            policy.update_q_entry(state, action, new_q_value);

            // set S = S'
            state = episode_step.state;

            if render_env {
                environment.render();
            }

            if episode_step.terminated | (step_number >= episode_max_len) {
                break;
            }
        }
    }
    Ok(())
}

/// Double Q-Learning, update rule
/// $$ Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha \left[ R + \gamma Q_1(S_{t+1}, argmax_a Q_2(S_{t+1}, a)) - Q(S_t, A_t) \right] $$
///
/// To prevent maximization bias, double q-learning uses two Q functions (or Q tabular policies)
///
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `episodes`: number of episodes to generate
/// - `episode_max_len`: maximum length of an episode
/// - `gamma`: discount factor
/// - `step_size`: step size for q update
///
/// ## Returns
///
/// `new_policy`: new policy that was optimized over the environment
#[allow(clippy::too_many_arguments)]
pub fn double_qlearning<P, E, R>(
    policy: &mut P,
    environment: &mut E,
    episodes: i32,
    episode_max_len: i32,
    gamma: f32,
    step_size: f32,
    render_env: bool,
    rng: &mut R,
) -> Result<(), LearningError>
where
    P: TabularPolicy + Clone,
    E: TabularEnvironment,
    R: Rng + ?Sized,
{
    // Q function initialization
    // Q table can have arbitrarily values except Q(terminal, •) which needs to be 0
    // therefore we set the value for each terminal state to 0
    for terminal_state in environment.get_terminal_states() {
        for i in 0..environment.get_number_actions() {
            policy.update_q_entry(terminal_state, i, 0.0);
        }
    }

    let mut policy_1 = policy.clone();
    let mut policy_2 = policy.clone();

    // loop for each episode
    for _ in 0..episodes {
        // init environment
        let mut state = environment.reset();

        let mut step_number = 0;
        // loop until is S is terminal
        loop {
            // Set q table of policy as the sum of the two tables
            policy.set_q(policy_1.get_q() + policy_2.get_q());

            // choose action from S with policy
            let action = policy.step(state, rng).map_err(LearningError::PolicyStep)?;

            step_number += 1;
            // take action A and observer R and S'
            let episode_step = environment
                .step(action, rng)
                .map_err(LearningError::EnvironmentStep)?;

            let (update_policy, max_policy) =
                if rand_distr::Bernoulli::new(0.5).unwrap().sample(rng) {
                    (&mut policy_1, &mut policy_2)
                } else {
                    (&mut policy_2, &mut policy_1)
                };
            // update q entry with Q(S, A) = Q(S, A) + step_size [ R + gamma * max_a Q(S', a) - Q(S, A) ]
            let q_sa = update_policy.get_q_value(state, action);
            let q_max = update_policy.get_q_value(
                episode_step.state,
                max_policy
                    .get_best_a(episode_step.state)
                    .map_err(LearningError::PolicyStep)?,
            );
            let new_q_value = q_sa + step_size * (episode_step.reward + gamma * q_max - q_sa);
            update_policy.update_q_entry(state, action, new_q_value);

            // set S = S'
            state = episode_step.state;

            if render_env {
                environment.render();
            }

            if episode_step.terminated | (step_number >= episode_max_len) {
                break;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {}
