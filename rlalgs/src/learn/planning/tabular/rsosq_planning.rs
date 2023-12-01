//! Random sample one step tabular Q planning algorithm

use indicatif::{ProgressBar, ProgressIterator};
use ndarray::Array2;
use rand::Rng;

use crate::{
    learn::{LearningError, VerbosityConfig},
    model::Model,
    policy::{Policy, ValuePolicy},
};

/// Parameters for random-sample one-step tabular q-planning algorithm
///
/// `n_iterations`: number of sampling-update iterations to do
/// `gamma`: discount factor
/// `step_size`: step size of the update rule
pub struct Params {
    pub n_iterations: i32,
    pub gamma: f32,
    pub step_size: f32,
}

/// Random Sample one step tabular Q-planning
/// $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right] $$
///
/// This simple planning tabular algorithm at each iteration samples a state, action pair from
/// the model and obtains next state and reward. Then uses this information to update
/// the Q value using Q-Learning update rule.
///
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `params`: algorithm parameters
/// - `rng`: random generator
/// - `verbosity`: verbosity configuration
///
pub fn learn<P, M, R>(
    mut policy: P,
    model: &M,
    params: Params,
    rng: &mut R,
    verbosity: &VerbosityConfig,
) -> Result<P, LearningError<i32, i32>>
where
    P: Policy<State = i32, Action = i32>
        + ValuePolicy<State = i32, Action = i32, Q = Array2<f32>, Update = f32>,
    R: Rng + ?Sized,
    M: Model<State = i32, Action = i32>,
{
    let progress_bar = if verbosity.learning_progress {
        ProgressBar::new(params.n_iterations as u64)
    } else {
        ProgressBar::hidden()
    };

    for _ in (0..params.n_iterations).progress_with(progress_bar) {
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
        policy.update_q_entry(&state, &action, &new_value);
    }

    Ok(policy)
}
