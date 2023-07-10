use crate::TabularStateAction;
use indicatif::{ProgressBar, ProgressIterator};
use keyed_priority_queue::KeyedPriorityQueue;
use ordered_float::OrderedFloat;
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
/// - `tolerance`: minimum value of the update no put state, action pair into the priority queue
/// - `gamma`: discount factor
/// - `step_size`: step size of the update rule
pub struct Params {
    pub n_iterations: i32,
    pub real_world_steps: i32,
    pub simulation_steps: i32,
    pub tolerance: f32,
    pub gamma: f32,
    pub step_size: f32,
}

/// Compute policy update
fn compute_update<P>(
    policy: &P,
    state_action: TabularStateAction,
    next_state: i32,
    reward: f32,
    params: &Params,
) -> Result<f32, LearningError>
where
    P: TabularPolicy,
{
    let q_sa = policy.get_q_value(state_action.state, state_action.action);
    let q_max = policy
        .get_max_q_value(next_state)
        .map_err(LearningError::PolicyStep)?;
    Ok(reward + params.gamma * q_max - q_sa)
}

/// if the requirement is met then the state, action pair is put on queue
fn put_on_queue(
    state_action: TabularStateAction,
    update: f32,
    priority_queue: &mut KeyedPriorityQueue<TabularStateAction, OrderedFloat<f32>>,
    params: &Params,
) {
    if update.abs() > params.tolerance {
        priority_queue.push(state_action, OrderedFloat::from(update.abs()));
    }
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

    // let priority queue
    let mut priority_queue: KeyedPriorityQueue<TabularStateAction, OrderedFloat<f32>> =
        KeyedPriorityQueue::new();

    // init environment
    let mut state = environment.reset();

    for _ in (0..params.n_iterations).progress_with(progress_bar) {
        // choose action from S with policy
        let action = policy.step(state, rng).map_err(LearningError::PolicyStep)?;
        let episode_step = environment
            .step(action, rng)
            .map_err(LearningError::EnvironmentStep)?;

        // Direct Learning
        let update = compute_update(
            &policy,
            TabularStateAction { state, action },
            episode_step.state,
            episode_step.reward,
            &params,
        )?;
        put_on_queue(
            TabularStateAction { state, action },
            update,
            &mut priority_queue,
            &params,
        );

        // update model
        model.update_step(state, action, episode_step.state, episode_step.reward);

        // Planning

        for _ in 0..params.simulation_steps {
            if priority_queue.is_empty() {
                break;
            }
            let model_sa = priority_queue.pop().unwrap().0;
            let step = model.predict_step(model_sa.state, model_sa.action);

            let q_sa = policy.get_q_value(model_sa.state, model_sa.action);
            let q_max = policy
                .get_max_q_value(model_sa.state)
                .map_err(LearningError::PolicyStep)?;
            policy.update_q_entry(
                model_sa.state,
                model_sa.action,
                q_sa + params.step_size
                    * compute_update(&policy, model_sa, step.state, step.reward, &params)?,
            );

            // loop for all S,A predicted to lead to S
            for preceding_sa in model.get_preceding_sa(model_sa.state) {
                let step = model.predict_step(preceding_sa.state, preceding_sa.action);
                let update =
                    compute_update(&policy, preceding_sa, step.state, step.reward, &params)?;
                put_on_queue(preceding_sa, update, &mut priority_queue, &params);
            }
        }

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

    Ok((policy, model))
}
