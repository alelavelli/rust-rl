use indicatif::{ProgressBar, ProgressIterator};
use ndarray::Array2;
use rand::Rng;
use rlenv::{tabular::TabularEnvironment, Environment};

use crate::{
    learn::{LearningError, VerbosityConfig},
    model::Model,
    policy::{Policy, ValuePolicy},
};

/// Parameters for dyna-q learning algorithm
///
/// - `n_iterations`: number of iteration of the algorithm.
/// An iteration consists of making a real environment step
/// and n simulated steps
/// - `real_world_steps`: number of real world steps at each iteration
/// - `simulation_steps`: number of simulation steps at each iteration
/// - `gamma`: discount factor
/// - `step_size`: step size of the update rule
pub struct Params {
    pub n_iterations: i32,
    pub real_world_steps: i32,
    pub simulation_steps: i32,
    pub gamma: f32,
    pub step_size: f32,
}

/// Dyna-Q model based learning algorithm
/// $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right] $$
///
/// In each iteration of Dyna-Q algorithm, the agent takes one real environment step
/// and then n simulated steps. The policy is updated at each step with Q learning algorithm
///
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `model`: TabularModel
/// - `params`: algorithm parameters
/// - `rng`: random generator
/// - `verbosity`: verbosity configuration
pub fn learn<P, E, R, M>(
    mut policy: P,
    mut environment: E,
    mut model: M,
    params: Params,
    rng: &mut R,
    verbosity: &VerbosityConfig,
) -> Result<(P, M), LearningError<i32, i32>>
where
    P: Policy<State = i32, Action = i32> + ValuePolicy<State = i32, Action = i32, Q = Array2<f32>>,
    E: Environment<State = i32, Action = i32> + TabularEnvironment,
    R: Rng + ?Sized,
    M: Model<State = i32, Action = i32>,
{
    let progress_bar = if verbosity.learning_progress {
        ProgressBar::new(params.n_iterations as u64)
    } else {
        ProgressBar::hidden()
    };

    // init environment
    let mut state = environment.reset();

    for _ in (0..params.n_iterations).progress_with(progress_bar) {
        for _ in 0..params.real_world_steps {
            // choose action from S with policy
            let action = policy
                .step(&state, rng)
                .map_err(|err| LearningError::PolicyStep { source: err, state })?;
            let episode_step =
                environment
                    .step(&action, rng)
                    .map_err(|err| LearningError::EnvironmentStep {
                        source: err,
                        action,
                    })?;

            // Direct Learning
            let q_sa = policy.get_q_value(&state, &action);
            let q_max = policy
                .get_max_q_value(&episode_step.next_state)
                .map_err(|err| LearningError::PolicyStep {
                    source: err,
                    state: episode_step.next_state,
                })?;
            let new_value =
                q_sa + params.step_size * (episode_step.reward + params.gamma * q_max - q_sa);
            policy.update_q_entry(&state, &action, new_value);

            // update model
            model.update_step(
                &state,
                &action,
                &episode_step.next_state,
                episode_step.reward,
            );

            if episode_step.terminated {
                // if we reached terminal state we reset the environment
                state = environment.reset();
            } else {
                state = episode_step.next_state;
            }

            if verbosity.render_env {
                environment.render();
            }
        }

        // Planning
        for _ in 0..params.simulation_steps {
            // 1 select a state and an action at random
            let (state, action) = if let Some(sa_sample) = model.sample_sa(rng) {
                (sa_sample.state, sa_sample.action)
            } else {
                return Err(LearningError::ModelError);
            };

            // 2 get next state and reward
            let next_step_sample = model.predict_step(&state, &action);

            // 3 appy one-step tabular Q-learning
            let q_sa = policy.get_q_value(&state, &action);
            let q_max = policy
                .get_max_q_value(&next_step_sample.state)
                .map_err(|err| LearningError::PolicyStep {
                    source: err,
                    state: next_step_sample.state,
                })?;
            let new_value =
                q_sa + params.step_size * (next_step_sample.reward + params.gamma * q_max - q_sa);
            policy.update_q_entry(&state, &action, new_value);
        }
    }

    Ok((policy, model))
}
