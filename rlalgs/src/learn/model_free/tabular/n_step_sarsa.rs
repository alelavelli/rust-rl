use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use ndarray::Array2;
use rand::Rng;
use rand_distr::num_traits::Pow;
use rlenv::{tabular::TabularEnvironment, Environment};
use std::cmp::min;

use crate::{
    learn::{LearningError, VerbosityConfig},
    policy::{Policy, ValuePolicy},
};

/// Parameters for sarsa learning algorithm
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

/// N-step Sarsa,
/// $$ G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}), n \ge 1, 0 \le t \le T - n $$ update rule:
/// $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ G_{t:t+n} - Q(S_t, A_t) \right] $$
///
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `params`: algorithm parameters
/// - `rng`: random generator
/// - `verbosity`: verbosity configuration
pub fn learn<P, E, R>(
    mut policy: P,
    mut environment: E,
    params: Params,
    rng: &mut R,
    versbosity: &VerbosityConfig,
) -> Result<P, LearningError>
where
    P: Policy<State = i32, Action = i32> + ValuePolicy<State = i32, Action = i32, Q = Array2<f32>>,
    E: Environment<State = i32, Action = i32> + TabularEnvironment,
    R: Rng + ?Sized,
{
    let n = params.n as f32;

    let multiprogress_bar = MultiProgress::new();
    let progress_bar = multiprogress_bar.add(ProgressBar::new(params.episodes as u64));

    for _ in (0..params.episodes).progress_with(progress_bar) {
        let mut states: Vec<i32> = Vec::new();
        let mut actions: Vec<i32> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();

        // init environment and store it
        let state = environment.reset();
        states.push(state);

        if versbosity.render_env {
            environment.render();
        }

        // choose action from S with policy and store it
        let action = policy
            .step(&state, rng)
            .map_err(LearningError::PolicyStep)?;
        actions.push(action);

        let mut capital_t = std::f32::INFINITY;
        let mut t: f32 = 0.0;
        // loop until is S is terminal
        loop {
            if t < capital_t {
                // take action A and observer R and S'
                let episode_step = environment
                    .step(&actions[t as usize], rng)
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
                    let next_action = policy
                        .step(&episode_step.next_state, rng)
                        .map_err(LearningError::PolicyStep)?;
                    actions.push(next_action);
                }
            }
            // set tau as the time whose estimate is being updated
            let tau = t - n + 1.0;
            if tau >= 0.0 {
                // compute the return as the discounted sum of rewards
                let mut return_g = 0.0;
                for i in (tau as i32 + 1)..=min((tau + n) as i32, capital_t as i32) {
                    // we access to rewards with index (i - 1) because the reward at step 0 is missing
                    // therefore, position 0 contains reward of step 1
                    return_g += params.gamma.pow(i - tau as i32 - 1) * rewards[(i - 1) as usize];
                }
                if tau + n < capital_t {
                    if params.expected {
                        return_g += params.gamma.pow(n)
                            * policy.expected_q_value(&states[(tau + n) as usize]);
                    } else {
                        return_g += params.gamma.pow(n)
                            * policy.get_q_value(
                                &states[(tau + n) as usize],
                                &actions[(tau + n) as usize],
                            );
                    }
                }
                // update policy q at states and actions at time tau
                let q_sa_tau = policy.get_q_value(&states[tau as usize], &actions[tau as usize]);
                let new_q_value = q_sa_tau + params.step_size * (return_g - q_sa_tau);
                policy.update_q_entry(&states[tau as usize], &actions[tau as usize], new_q_value);
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
