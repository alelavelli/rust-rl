pub mod montecarlo;

use rlenv::tabular::{TabularEnvironment, TabularEpisode};

use crate::{policy::tabular::TabularPolicy, EpisodeGenerationError};

/// generate episode using policy and environment
///
/// ## Parameters
///
/// `policy`: policy to use in the environment
/// `environment`: environment to use
/// `rng`: random seed
fn generate_tabular_episode<P, E>(
    policy: &mut P,
    environment: &mut E,
    rng: &mut rand::rngs::ThreadRng,
) -> Result<TabularEpisode, EpisodeGenerationError>
where
    P: TabularPolicy,
    E: TabularEnvironment,
{
    // trace variables
    let mut states: Vec<i32> = Vec::new();
    let mut actions: Vec<i32> = Vec::new();
    let mut rewards: Vec<f32> = Vec::new();
    // get the initial state
    let mut state = environment.reset();
    let mut action: i32;
    let mut reward: f32;

    // loop to generate the episode
    // Monte Carlo only works for terminating environments, hence, we do not need to set maximum episode length
    while !environment.is_terminal(state) {
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
        state = episode_step.observation;
        reward = episode_step.reward;

        // record r_{t+1}
        rewards.push(reward);
    }
    Ok(TabularEpisode {
        states,
        actions,
        rewards,
    })
}
