use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use ndarray::Array2;
use rand::Rng;
use rlenv::{tabular::TabularEnvironment, Environment};
use std::cmp::min;

use crate::{
    learn::{LearningError, VerbosityConfig},
    policy::{Policy, ValuePolicy},
};

pub struct SigmaInput {
    pub state: i32,
    pub action: i32,
    pub step: i32,
}

/// Parameters for n step q sigma learning algorithm
///
/// `episodes`: number of episodes to generate during learning
/// `gamma`: discount factor
/// `step_size`: step size of the update rule
/// `n`: number of steps before compute the update
/// `sigma_fn`: function that takes as input state, action at time t and
/// returns the value of sigma that represents the degree of sampling: 1 for full sampling and 0 for pure expectation
/// `update_behaviour`: true to update the behaviour poilcy Q with the learnt one in order to speed up the learning process.
/// The behaviour policy remains highly explorative with respect to the policy to learn.
pub struct Params {
    pub episodes: i32,
    pub gamma: f32,
    pub step_size: f32,
    pub n: i32,
    pub sigma_fn: Box<dyn Fn(SigmaInput) -> f32>,
    pub update_behaviour: bool,
}

/// n-step q sigma learning
/// $$ G_{t:h} = R_{t+1} + \gamma \left( \sigma_{t+1}\rho_{t+1}+(1-\sigma_{t+1})\pi(A_{t+1}, S_{t+1}) \right)\left( G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1}) \right) + \gamma \bar{V}_{h-1}(S_{t+1}) $$ update rule:
/// $$ Q_{t+n}(S_t, A_t) \leftarrow Q_{t+n-1}(S_t, A_t) + \alpha \left[ G_{t:t+n} - Q_{t+n-1}(S_t, A_t) \right] $$
///
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `params`: algorithm parameters
/// - `rng`: random generator
/// - `verbosity`: verbosity configuration
pub fn learn<P, B, E, R>(
    mut policy: P,
    mut behaviour: B,
    mut environment: E,
    params: Params,
    rng: &mut R,
    versbosity: &VerbosityConfig,
) -> Result<P, LearningError>
where
    P: Policy<State = i32, Action = i32> + ValuePolicy<State = i32, Action = i32, Q = Array2<f32>>,
    B: Policy<State = i32, Action = i32> + ValuePolicy<State = i32, Action = i32, Q = Array2<f32>>,
    E: Environment<State = i32, Action = i32> + TabularEnvironment,
    R: Rng + ?Sized,
{
    let n = params.n as f32;

    let multiprogress_bar = MultiProgress::new();
    let progress_bar = multiprogress_bar.add(ProgressBar::new(params.episodes as u64));

    for _ in (0..params.episodes).progress_with(progress_bar) {
        if params.update_behaviour {
            behaviour.set_q(policy.get_q().clone());
        }

        let mut states: Vec<i32> = Vec::new();
        let mut actions: Vec<i32> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut sigmas: Vec<f32> = Vec::new();
        let mut rhos: Vec<f32> = Vec::new();

        // init environment and store it
        let state = environment.reset();
        states.push(state);

        if versbosity.render_env {
            environment.render();
        }

        // choose action from S with policy and store it
        let action = behaviour
            .step(state, rng)
            .map_err(LearningError::PolicyStep)?;
        actions.push(action);

        let mut capital_t = std::f32::INFINITY;
        let mut t: f32 = 0.0;
        // loop until is S is terminal
        loop {
            if t < capital_t {
                // take action A and observer R and S'
                let episode_step = environment
                    .step(actions[t as usize], rng)
                    .map_err(LearningError::EnvironmentStep)?;
                // store next state and reward
                states.push(episode_step.next_state);
                rewards.push(episode_step.reward);
                // if the episode is terminated then set capital T to the next value of t so that
                // we update the policy Q
                if episode_step.terminated {
                    capital_t = t + 1.0;
                } else {
                    // otherwise do the next step
                    let next_action = behaviour
                        .step(episode_step.next_state, rng)
                        .map_err(LearningError::PolicyStep)?;
                    actions.push(next_action);

                    // select and store sigma t + 1
                    sigmas.push(params.sigma_fn.as_ref()(SigmaInput {
                        state: episode_step.next_state,
                        action: next_action,
                        step: t as i32,
                    }));

                    // store importance retio
                    rhos.push(
                        policy.action_prob(episode_step.next_state, next_action)
                            / behaviour.action_prob(episode_step.next_state, next_action),
                    );
                }
            }
            // set tau as the time whose estimate is being updated
            let tau = t - n + 1.0;
            if tau >= 0.0 {
                // compute the return as the discounted sum of rewards
                let mut return_g = 0.0;

                if (t + 1.0) < capital_t {
                    return_g = policy.get_q_value(states[t as usize + 1], actions[t as usize + 1]);
                }

                for k in ((tau as i32 + 1)..=min(t as i32 + 1, capital_t as i32)).rev() {
                    if k as f32 == capital_t {
                        // we access to index - 1 because reward at step 0 is missing
                        return_g = rewards[capital_t as usize - 1];
                    } else {
                        let k_idx = k as usize;
                        let state_k = states[k_idx];
                        let action_k = actions[k_idx];
                        let sigma_k = sigmas[k_idx - 1];
                        let rho_k = rhos[k_idx - 1];
                        let q_sa_k = policy.get_q_value(state_k, action_k);
                        let a_k_prob = policy.action_prob(state_k, action_k);

                        let state_value = policy.expected_q_value(state_k);
                        return_g = rewards[k as usize - 1]
                            + params.gamma
                                * (sigma_k * rho_k + (1.0 - sigma_k) * a_k_prob)
                                * (return_g - q_sa_k)
                            + params.gamma * state_value;
                    }
                }

                // update policy q at states and actions at time tau
                let q_sa_tau = policy.get_q_value(states[tau as usize], actions[tau as usize]);
                let new_q_value = q_sa_tau + params.step_size * (return_g - q_sa_tau);

                policy.update_q_entry(states[tau as usize], actions[tau as usize], new_q_value);
            }

            if versbosity.render_env {
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
