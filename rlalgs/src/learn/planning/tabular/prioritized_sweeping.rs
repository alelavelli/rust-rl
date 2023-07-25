use indicatif::{ProgressBar, ProgressIterator};
use keyed_priority_queue::KeyedPriorityQueue;
use ndarray::Array2;
use ordered_float::OrderedFloat;
use rand::Rng;
use rlenv::{tabular::TabularEnvironment, Environment};

use crate::{
    learn::{LearningError, VerbosityConfig},
    model::Model,
    policy::{Policy, ValuePolicy},
    StateAction,
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
    pub simulation_steps: i32,
    pub tolerance: f32,
    pub gamma: f32,
    pub step_size: f32,
}

/// Compute policy update
fn compute_update<P>(
    policy: &P,
    state_action: &StateAction<i32, i32>,
    next_state: i32,
    reward: f32,
    params: &Params,
) -> Result<f32, LearningError>
where
    P: Policy<i32, i32> + ValuePolicy<i32, i32, Array2<f32>>,
{
    let q_sa = policy.get_q_value(state_action.state, state_action.action);
    let q_max = policy
        .get_max_q_value(next_state)
        .map_err(LearningError::PolicyStep)?;
    Ok(reward + params.gamma * q_max - q_sa)
}

/// if the requirement is met then the state, action pair is put on queue
fn put_on_queue(
    state_action: StateAction<i32, i32>,
    update: f32,
    priority_queue: &mut KeyedPriorityQueue<StateAction<i32, i32>, OrderedFloat<f32>>,
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
    P: Policy<i32, i32> + ValuePolicy<i32, i32, Array2<f32>>,
    E: Environment<i32, i32> + TabularEnvironment,
    R: Rng + ?Sized,
    M: Model<i32, i32>,
{
    let progress_bar = ProgressBar::new(params.n_iterations as u64);

    // let priority queue
    let mut priority_queue: KeyedPriorityQueue<StateAction<i32, i32>, OrderedFloat<f32>> =
        KeyedPriorityQueue::new();

    // init environment
    let mut state = environment.reset();

    for _ in (0..params.n_iterations).progress_with(progress_bar) {
        // choose action from S with policy
        let action = policy.step(state, rng).map_err(LearningError::PolicyStep)?;
        let episode_step = environment
            .step(action, rng)
            .map_err(LearningError::EnvironmentStep)?;
        // update model
        model.update_step(state, action, episode_step.next_state, episode_step.reward);

        // Direct Learning
        let update = compute_update(
            &policy,
            &StateAction { state, action },
            episode_step.next_state,
            episode_step.reward,
            &params,
        )?;
        put_on_queue(
            StateAction { state, action },
            update,
            &mut priority_queue,
            &params,
        );

        // Planning

        for _ in 0..params.simulation_steps {
            if priority_queue.is_empty() {
                //println!("I'm out at {i}-{n}");
                break;
            }
            let queue_sa = priority_queue.pop().unwrap().0;
            let model_step = model.predict_step(queue_sa.state, queue_sa.action);

            let q_sa = policy.get_q_value(queue_sa.state, queue_sa.action);
            policy.update_q_entry(
                queue_sa.state,
                queue_sa.action,
                q_sa + params.step_size
                    * compute_update(
                        &policy,
                        &queue_sa,
                        model_step.state,
                        model_step.reward,
                        &params,
                    )?,
            );

            // loop for all S,A predicted to lead to S
            for preceding_sa in model.get_preceding_sa(queue_sa.state).unwrap() {
                let step = model.predict_step(preceding_sa.state, preceding_sa.action);
                let update =
                    compute_update(&policy, preceding_sa, step.state, step.reward, &params)?;
                put_on_queue(*preceding_sa, update, &mut priority_queue, &params);
            }
        }

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

    Ok((policy, model))
}
