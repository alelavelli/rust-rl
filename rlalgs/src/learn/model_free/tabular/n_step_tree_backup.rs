use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use ndarray::Array2;
use rand::Rng;
use rlenv::{tabular::TabularEnvironment, Environment};
use std::cmp::min;

use crate::{
    learn::{LearningError, VerbosityConfig},
    policy::{Policy, ValuePolicy},
};

/// Parameters for n step tree backup learning algorithm
///
/// `episodes`: number of episodes to generate during learning
/// `gamma`: discount factor
/// `step_size`: step size of the update rule
/// `expeced`: true to use expected sarsa instead of standard one
/// `n`: number of steps before compute the update
pub struct Params {
    pub episodes: i32,
    pub gamma: f32,
    pub step_size: f32,
    pub expected: bool,
    pub n: i32,
}

/// n-step tree backup
/// $$ G_{t:t+n} = R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a|S_{t+1})Q_{t+n-1}(S_{t+1}, a) + \gamma \pi(A_{t+1}|S_{t+1})G_{t+1:t+n} $$ update rule:
/// $$ Q_{t+n}(S_t, A_t) \leftarrow Q_{t+n-1}(S_t, A_t) + \alpha \left[ G_{t:t+n} - Q_{t+n-1}(S_t, A_t) \right] $$
///
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `params`: algorithm parameters
/// - `rng`: random generator
/// - `verbosity`: verbosity configuration
#[allow(clippy::too_many_arguments)]
pub fn learn<P, E, R>(
    mut policy: P,
    mut environment: E,
    params: Params,
    rng: &mut R,
    verbosity: &VerbosityConfig,
) -> Result<P, LearningError<i32, i32>>
where
    P: Policy<State = i32, Action = i32> + ValuePolicy<State = i32, Action = i32, Q = Array2<f32>>,
    E: Environment<State = i32, Action = i32> + TabularEnvironment,
    R: Rng + ?Sized,
{
    let n = params.n as f32;

    let multiprogress_bar = MultiProgress::new();
    let progress_bar = if verbosity.learning_progress {
        multiprogress_bar.add(ProgressBar::new(params.episodes as u64))
    } else {
        multiprogress_bar.add(ProgressBar::hidden())
    };

    for _ in (0..params.episodes).progress_with(progress_bar) {
        let mut states: Vec<i32> = Vec::new();
        let mut actions: Vec<i32> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();

        // init environment and store it
        let state = environment.reset();
        states.push(state);

        if verbosity.render_env {
            environment.render();
        }

        // choose action from S with policy and store it
        let action = policy
            .step(&state, rng)
            .map_err(|err| LearningError::PolicyStep { source: err, state })?;
        actions.push(action);

        let mut capital_t = std::f32::INFINITY;
        let mut t: f32 = 0.0;
        // loop until is S is terminal
        loop {
            if t < capital_t {
                // take action A and observer R and S'
                let episode_step = environment.step(&actions[t as usize], rng).map_err(|err| {
                    LearningError::EnvironmentStep {
                        source: err,
                        action: actions[t as usize],
                    }
                })?;
                // store next state and reward
                states.push(episode_step.next_state);
                rewards.push(episode_step.reward);
                // if the episode is terminated then set capital T to the next value of t so that
                // we update the policy Q
                if episode_step.terminated {
                    capital_t = t + 1.0;
                } else {
                    // otherwise do the next step
                    let next_action =
                        policy.step(&episode_step.next_state, rng).map_err(|err| {
                            LearningError::PolicyStep {
                                source: err,
                                state: episode_step.next_state,
                            }
                        })?;
                    actions.push(next_action);
                }
            }
            // set tau as the time whose estimate is being updated
            let tau = t - n + 1.0;
            if tau >= 0.0 {
                // compute the return as the discounted sum of rewards
                let mut return_g;

                if (t + 1.0) >= capital_t {
                    // we access to reward with index decreased by 1 since the reward at step 0 is missing
                    return_g = rewards[capital_t as usize - 1];
                } else {
                    return_g = rewards[t as usize]
                        + params.gamma * policy.expected_q_value(&states[(t + 1.0) as usize]);
                }

                for k in ((tau as i32 + 1)..=min(t as i32, capital_t as i32 - 1)).rev() {
                    let k_idx = k as usize;
                    let state_k = states[k_idx];
                    let action_k = actions[k_idx];
                    let a_k_prob = policy.action_prob(&state_k, &action_k);

                    // here we add three terms:
                    // 1. retard at time k
                    // 2. q value with boostrapping for non chosen actions
                    // 3. return with chosen action
                    // we compute the second term by getting the expected q value and subtracting the chosen action contribution
                    let other_actions_contribution = params.gamma
                        * (policy.expected_q_value(&state_k)
                            - a_k_prob * policy.get_q_value(&state_k, &action_k));
                    let chosen_action_contribution = params.gamma * (a_k_prob * return_g);
                    return_g = rewards[k as usize - 1]
                        + other_actions_contribution
                        + chosen_action_contribution;
                }

                // update policy q at states and actions at time tau
                let q_sa_tau = policy.get_q_value(&states[tau as usize], &actions[tau as usize]);
                let new_q_value = q_sa_tau + params.step_size * (return_g - q_sa_tau);

                policy.update_q_entry(&states[tau as usize], &actions[tau as usize], new_q_value);
            }

            if verbosity.render_env {
                environment.render();
            }

            // loop until tau = T - 1
            if tau >= capital_t - 1.0 {
                break;
            }

            t += 1.0;
        }
    }
    Ok(policy)
}
