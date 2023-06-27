use rand::Rng;
use rand_distr::num_traits::Pow;
use rlenv::tabular::TabularEnvironment;
use std::cmp::min;

use crate::{learn::LearningError, policy::tabular::TabularPolicy};

/// N-step Sarsa,
/// $$ G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}), n \ge 1, 0 \le t \le T - n $$ update rule:
/// $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ G_{t:t+n} - Q(S_t, A_t) \right] $$
///
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `episodes`: number of episodes to generate
/// - `episode_max_len`: maximum length of an episode
/// - `n`: number of steps for the update rule
/// - `gamma`: discount factor
/// - `step_size`: step size for q update
/// - `render_env`: if true render the environment during the learning
/// - `rng`: random generator
#[allow(clippy::too_many_arguments)]
pub fn n_step_sarsa<P, E, R>(
    policy: &mut P,
    environment: &mut E,
    episodes: i32,
    n: i32,
    gamma: f32,
    step_size: f32,
    render_env: bool,
    rng: &mut R,
) -> Result<(), LearningError>
where
    P: TabularPolicy,
    E: TabularEnvironment,
    R: Rng + ?Sized,
{
    let n = n as f32;

    // loop for each episode
    for _ in 0..episodes {
        let mut states: Vec<i32> = Vec::new();
        let mut actions: Vec<i32> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();

        // init environment and store it
        let state = environment.reset();
        states.push(state);

        if render_env {
            environment.render();
        }

        // choose action from S with policy and store it
        let action = policy.step(state, rng).map_err(LearningError::PolicyStep)?;
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
                states.push(episode_step.state);
                rewards.push(episode_step.reward);
                // if the episode is terminated then set capital T to the next value of t so that
                // we update the policy Q
                if episode_step.terminated {
                    capital_t = t + 1.0;
                } else {
                    // otherwise do the next step
                    let next_action = policy
                        .step(episode_step.state, rng)
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
                    return_g += gamma.pow(i - tau as i32 - 1) * rewards[(i - 1) as usize];
                }
                if tau + n < capital_t {
                    return_g += gamma.pow(n)
                        * policy.get_q_value(
                            states[(tau + n) as usize],
                            actions[(tau + n) as usize],
                        );
                }
                // update policy q at states and actions at time tau
                let q_sa_tau = policy.get_q_value(states[tau as usize], actions[tau as usize]);
                let new_q_value = q_sa_tau + step_size * (return_g - q_sa_tau);
                policy.update_q_entry(states[tau as usize], actions[tau as usize], new_q_value);
                
            }

            if render_env {
                environment.render();
            }

            // loop until tau = T - 1
            if tau >= capital_t - 1.0 {
                break;
            }

            t += 1.0;
        }
    }
    Ok(())
}
