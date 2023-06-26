pub mod montecarlo;
pub mod temporal_difference;

use rand::Rng;
use rlenv::tabular::{TabularEnvironment, TabularEpisode};

use crate::{policy::tabular::TabularPolicy, EpisodeGenerationError};

/// generate episode using policy and environment
///
/// ## Parameters
///
/// `policy`: policy to use in the environment
/// `environment`: environment to use
/// `episode_max_len`: maximum number of steps in the episode
/// `rng`: random seed
pub fn generate_tabular_episode<P, E, R>(
    policy: &mut P,
    environment: &mut E,
    episode_max_len: Option<i32>,
    rng: &mut R,
    render_env: bool,
) -> Result<TabularEpisode, EpisodeGenerationError>
where
    P: TabularPolicy,
    E: TabularEnvironment,
    R: Rng + ?Sized,
{
    // trace variables
    let mut states: Vec<i32> = Vec::new();
    let mut actions: Vec<i32> = Vec::new();
    let mut rewards: Vec<f32> = Vec::new();
    // get the initial state
    let mut state = environment.reset();
    let mut action: i32;
    let mut reward: f32;

    if render_env {
        environment.render();
    }
    let episode_max_len = if let Some(value) = episode_max_len {
        value as f64
    } else {
        std::f64::INFINITY
    };

    let mut step_number = 0;
    // loop to generate the episode
    // Monte Carlo only works for terminating environments, hence, we do not need to set maximum episode length
    loop {
        step_number += 1;
        // get action from policy
        action = policy
            .step(state, rng)
            .map_err(EpisodeGenerationError::PolicyStep)?;
        // record s_t, a_t pair
        states.push(state);
        actions.push(action);
        // make environment step
        let episode_step = environment
            .step(action, rng)
            .map_err(EpisodeGenerationError::EnvironmentStep)?;
        state = episode_step.state;
        reward = episode_step.reward;

        // record r_{t+1}
        rewards.push(reward);

        if render_env {
            environment.render();
        }

        if episode_step.terminated | (step_number as f64 >= episode_max_len) {
            break;
        }
    }
    Ok(TabularEpisode {
        states,
        actions,
        rewards,
    })
}
