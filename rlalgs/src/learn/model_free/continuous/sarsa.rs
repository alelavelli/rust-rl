use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use ndarray::Array1;
use rand::Rng;
use rlenv::{continuous::DiscreteActionContinuousEnvironment, Environment};

use crate::{
    learn::{ContinuousLearningError, LearningError, VerbosityConfig},
    policy::{egreedy::ContinuousQ, DifferentiablePolicy, Policy, ValuePolicy},
};

/// Parameters for sarsa learning algorithm
///
/// `episodes`: number of episodes to generate during learning
/// `episode_max_len`: maximum length of a single episode before stopping it
/// `gamma`: discount factor
/// `step_size`: step size of the updating rule
pub struct Params {
    pub episodes: i32,
    pub episode_max_len: i32,
    pub gamma: f32,
    pub step_size: f32,
    pub expected: bool,
}

/// Sarsa on-policy TD with contunous state
///
/// Update rule:
/// $$ \boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha \left[  R + \gamma \hat{q}(S_{t+1}, A_{t+1}, \boldsymbol{w}) - \hat{q}(S_{t}, A_{t}, \boldsymbol{w}) \right] \nabla \hat{q}(S_{t}, A_{t}, \boldsymbol{w}) $$
pub fn learn<P, E, R>(
    mut policy: P,
    mut environment: E,
    params: Params,
    rng: &mut R,
    verbosity: &VerbosityConfig,
) -> Result<P, ContinuousLearningError>
where
    P: Policy<State = Array1<f32>, Action = i32>
        + ValuePolicy<State = Array1<f32>, Action = i32, Q = ContinuousQ, Update = Array1<f32>>
        + DifferentiablePolicy<State = Array1<f32>, Action = i32>,
    E: Environment<State = Array1<f32>, Action = i32> + DiscreteActionContinuousEnvironment,
    R: Rng + ?Sized,
{
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
            .map_err(|err| LearningError::PolicyStep {
                source: err,
                state: state.clone(),
            })?;

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
                todo!()
            } else {
                // choose A' from S' with policy
                let a_prime = policy.step(&episode_step.next_state, rng).map_err(|err| {
                    LearningError::PolicyStep {
                        source: err,
                        state: episode_step.next_state.clone(),
                    }
                })?;

                // differently from the tabular case, we cannot compute the new q value and setup it to the q function
                // instead, we compute the update factor and then we call the method so that the policy can update internally
                // the regressor
                let q_spap = policy.get_q_value(&episode_step.next_state, &a_prime);
                let weight_update = params.step_size
                    * (episode_step.reward + params.gamma * q_spap - q_sa)
                    * policy.gradient(&state, &action);
                // update q entry
                policy.update_q_entry(&state, &action, &weight_update);
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
