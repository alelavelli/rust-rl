use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use ndarray::Array2;
use rand::Rng;
use rand_distr::Distribution;
use rlenv::{tabular::TabularEnvironment, Environment};

use crate::{
    learn::{LearningError, VerbosityConfig}, policy::{Policy, ValuePolicy},
};
/// Double Q Learning parameters
///
/// `episodes`: number of episodes to generate during learning
/// `episode_max_len`: maximum length of a single episode before stopping it
/// `gamma`: discount factor
/// `step_size`: step size of the update rule
pub struct Params {
    pub episodes: i32,
    pub episode_max_len: i32,
    pub gamma: f32,
    pub step_size: f32,
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
/// - `params`: algorithm parameters
/// - `rng`: random generator
/// - `verbosity`: verbosity configuration
///
/// ## Returns
///
/// `new_policy`: new policy that was optimized over the environment
#[allow(clippy::too_many_arguments)]
pub fn learn<P, E, R>(
    mut policy: P,
    mut environment: E,
    params: Params,
    rng: &mut R,
    verbosity: &VerbosityConfig,
) -> Result<P, LearningError>
where
    P: Policy<i32, i32> + ValuePolicy<i32, i32, Array2<f32>> + Clone,
    E: Environment<i32, i32> + TabularEnvironment,
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

    let multiprogress_bar = MultiProgress::new();
    let progress_bar = multiprogress_bar.add(ProgressBar::new(params.episodes as u64));

    for _ in (0..params.episodes).progress_with(progress_bar) {
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
                episode_step.next_state,
                max_policy
                    .get_best_a(episode_step.next_state)
                    .map_err(LearningError::PolicyStep)?,
            );
            let new_q_value =
                q_sa + params.step_size * (episode_step.reward + params.gamma * q_max - q_sa);
            update_policy.update_q_entry(state, action, new_q_value);

            // set S = S'
            state = episode_step.next_state;

            if verbosity.render_env {
                environment.render();
            }

            if episode_step.terminated | (step_number >= params.episode_max_len) {
                break;
            }
        }
    }
    Ok(policy)
}

#[cfg(test)]
mod tests {}
