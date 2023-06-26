use ndarray::{s, Array1};
use rand::Rng;
use rlenv::tabular::TabularEnvironment;

use crate::{learn::LearningError, policy::tabular::TabularPolicy};

/// Sarsa: on-policy TD
///
/// ## Parameters
///
/// `policy`: TabularPolicy to learn
/// `environment`: TabularEnvironment
/// `episodes`: number of episodes to generate
/// `episode_max_len`: maximum length of an episode
/// `gamma`: discount factor
/// `step_size`: step size for q update
/// `expected`: true to use expected sarsa instead of standard sarsa learning algorithm
/// `render_env`: if true render the environment during the learning
/// `rng`: random generator
///
///
/// ## Expected Sarsa
///
/// Expected Sarsa is a off-policy learning algorithm similar to Q-learning that instead
/// of taking the maximum over next state-acation pair it uses the expected value, taking
/// into account how likely each action is under the current policy
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
            //println!("{}: {} -> {}", step_number, state, action);
            // take action A and observer R and S'
            let episode_step = environment
                .step(action, rng)
                .map_err(LearningError::EnvironmentStep)?;

            let q_sa = policy.get_q_value(state, action);

            if expected {
                // update q entry using weighted q value
                let a_probs = Array1::from(
                    (0..environment.get_number_actions())
                        .map(|i| policy.action_prob(episode_step.state, i))
                        .collect::<Vec<f32>>(),
                );
                let q_expected: f32 = policy
                    .get_q()
                    .slice(s![episode_step.state, ..])
                    .dot(&a_probs);

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
                //println!("steps {step_number}");
                //println!();
                break;
            }
        }
    }
    Ok(())
}

/// Q-Learning
///
/// Off-policy TD control algorithm
/// ## Parameters
///
/// `policy`: TabularPolicy to learn
/// `environment`: TabularEnvironment
/// `episodes`: number of episodes to generate
/// `episode_max_len`: maximum length of an episode
/// `gamma`: discount factor
/// `step_size`: step size for q update
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

#[cfg(test)]
mod tests {}
