use crate::learn::planning::tabular::rsosq_planning;
use indicatif::{ProgressBar, ProgressIterator};
use rand::Rng;
use rlenv::tabular::TabularEnvironment;

use crate::{
    learn::{LearningError, VerbosityConfig},
    model::tabular::TabularModel,
    policy::tabular::TabularPolicy,
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
#[allow(clippy::too_many_arguments)]
pub fn learn<P, E, R, M>(
    mut policy: P,
    mut environment: E,
    mut model: M,
    params: Params,
    rng: &mut R,
    verbosity: &VerbosityConfig,
) -> Result<(P, M), LearningError>
where
    P: TabularPolicy,
    E: TabularEnvironment,
    R: Rng + ?Sized,
    M: TabularModel,
{
    let progress_bar = ProgressBar::new(params.n_iterations as u64);

    // init environment
    let mut state = environment.reset();

    for _ in (0..params.n_iterations).progress_with(progress_bar) {
        for _ in 0..params.real_world_steps {
            // choose action from S with policy
            let action = policy.step(state, rng).map_err(LearningError::PolicyStep)?;
            let episode_step = environment
                .step(action, rng)
                .map_err(LearningError::EnvironmentStep)?;

            // Direct Learning
            let q_sa = policy.get_q_value(state, action);
            let q_max = policy
                .get_max_q_value(episode_step.state)
                .map_err(LearningError::PolicyStep)?;
            let new_value =
                q_sa + params.step_size * (episode_step.reward + params.gamma * q_max - q_sa);
            policy.update_q_entry(state, action, new_value);

            // update model
            model.update_step(state, action, episode_step.state, episode_step.reward);

            if episode_step.terminated {
                // if we reached terminal state we reset the environment
                state = environment.reset();
            } else {
                state = episode_step.state;
            }

            if verbosity.render_env {
                environment.render();
            }
        }
        // Planning
        let planning_params = rsosq_planning::Params {
            n_iterations: params.simulation_steps,
            gamma: params.gamma,
            step_size: params.step_size,
        };
        policy = rsosq_planning::learn(policy, &model, planning_params, rng, verbosity)?;
    }

    Ok((policy, model))
}
