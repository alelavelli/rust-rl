use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use ndarray::Array2;
use rand::Rng;
use rlenv::{tabular::TabularEnvironment, Environment};

use crate::{
    learn::{LearningError, VerbosityConfig},
    policy::{Policy, ValuePolicy},
};

/// Parameters for sarsa learning algorithm
///
/// `episodes`: number of episodes to generate during learning
/// `episode_max_len`: maximum length of a single episode before stopping it
/// `gamma`: discount factor
/// `step_size`: step size of the update rule
/// `expeced`: true to use expected sarsa instead of standard one
pub struct Params {
    pub episodes: i32,
    pub episode_max_len: i32,
    pub gamma: f32,
    pub step_size: f32,
    pub expected: bool,
}

/// Sarsa: on-policy TD, update rule:
/// $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right] $$
///
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `params`: algorithm parameters
/// - `rng`: random generator
/// - `verbosity`: verbosity configuration
///
///
/// ## Expected Sarsa
///
/// Expected Sarsa is a off-policy learning algorithm similar to Q-learning that instead
/// of taking the maximum over next state-acation pair it uses the expected value, taking
/// into account how likely each action is under the current policy
/// $$ Q(S, A) \leftarrow Q(S_t, A_t) + \alpha \left[ R + \gamma \sum_a \pi(a | S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right] $$
///
pub fn learn<P, E, R>(
    mut policy: P,
    mut environment: E,
    params: Params,
    rng: &mut R,
    verbosity: &VerbosityConfig,
) -> Result<P, LearningError<i32, i32>>
where
    P: Policy<State = i32, Action = i32> + ValuePolicy<State = i32, Action = i32, Q = Array2<f32>>,
    E: Environment<State = i32, Action = i32> + TabularEnvironment<State = i32>,
    R: Rng + ?Sized,
{
    // Q function initialization
    // Q table can have arbitrarily values except Q(terminal, •) which needs to be 0
    // therefore we set the value for each terminal state to 0
    for terminal_state in environment.get_terminal_states() {
        for i in 0..environment.get_number_actions() {
            policy.update_q_entry(&terminal_state, &i, 0.0);
        }
    }
    let multiprogress_bar = MultiProgress::new();
    let progress_bar = if verbosity.learning_progress {
        multiprogress_bar.add(ProgressBar::new(params.episodes as u64))
    } else {
        multiprogress_bar.add(ProgressBar::hidden())
    };

    for _ in (0..params.episodes).progress_with(progress_bar) {
        // init environment
        let mut state = environment.reset(rng);
        // choose action from S with policy
        let mut action = policy
            .step(&state, rng)
            .map_err(|err| LearningError::PolicyStep { source: err, state })?;

        let mut step_number = 0;
        // loop until is S is terminal
        loop {
            step_number += 1;
            // take action A and observer R and S'
            let episode_step =
                environment
                    .step(&action, rng)
                    .map_err(|err| LearningError::EnvironmentStep {
                        source: err,
                        action,
                    })?;

            let q_sa = policy.get_q_value(&state, &action);

            if params.expected {
                let q_expected = policy.expected_q_value(&episode_step.next_state);

                let new_q_value = q_sa
                    + params.step_size * (episode_step.reward + params.gamma * q_expected - q_sa);
                // update q entry
                policy.update_q_entry(&state, &action, new_q_value);

                action = policy.step(&episode_step.next_state, rng).map_err(|err| {
                    LearningError::PolicyStep {
                        source: err,
                        state: episode_step.next_state,
                    }
                })?;
            } else {
                // choose A' from S' with policy
                let a_prime = policy.step(&episode_step.next_state, rng).map_err(|err| {
                    LearningError::PolicyStep {
                        source: err,
                        state: episode_step.next_state,
                    }
                })?;

                // update q entry with Q(S, A) = Q(S, A) + step_size [ R + gamma * Q(S', A') - Q(S, A) ]
                let q_spap = policy.get_q_value(&episode_step.next_state, &a_prime);
                let new_q_value =
                    q_sa + params.step_size * (episode_step.reward + params.gamma * q_spap - q_sa);
                // update q entry
                policy.update_q_entry(&state, &action, new_q_value);
                // set A = A'
                action = a_prime;
            }

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
