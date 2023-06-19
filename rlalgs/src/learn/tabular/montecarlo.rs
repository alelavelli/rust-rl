use std::collections::HashMap;

use rlenv::tabular::TabularEnvironment;

use crate::{
    learn::{generate_episode, LearningError},
    policy::tabular::TabularPolicy,
};

/// Monte Carlo First Visit
///
/// ## Parameters
///
/// `policy`: TabularPolicy to learn
/// `environment`: TabularEnvironment
/// `episodes`: number of episodes to generate
/// `episode_max_len`: maximum length of an episode
/// `gamma`: discount factor
///
/// ## Returns
///
/// `new_policy`: new policy that was optimized over the environment
pub fn montecarlo<P, E>(
    policy: &mut P,
    environment: &mut E,
    episodes: i32,
    gamma: f32,
    first_visit: bool,
) -> Result<(), LearningError>
where
    P: TabularPolicy,
    E: TabularEnvironment,
{
    let mut rng = rand::thread_rng();

    // initialize variables
    // TODO: check if there is no entry for the state-action pair. In this case, use entry default to 0.0
    let mut returns: HashMap<(i32, i32), Vec<f32>> = HashMap::new();

    for _ in 0..episodes {
        // GENERATE EPISODE
        let episode = generate_episode(policy, environment, &mut rng)
            .map_err(LearningError::EpisodeGeneration)?;
        let (states, actions, rewards) = (episode.states, episode.actions, episode.rewards);
        // Update Q function
        let mut g = 0.0;
        for t in (0..states.len()).rev() {
            g = gamma * g + rewards[t];
            // If we are in first_visit settings then we check that the pair s,a a time t is the first visit
            // otherwise, we enter always
            if !first_visit | is_first_visit(states[t], actions[t], &states, &actions, t) {
                let sa_returns = returns.entry((states[t], actions[t])).or_insert(Vec::new());
                sa_returns.push(g);
                let new_q_value: f32 = sa_returns.iter().sum::<f32>() / sa_returns.len() as f32;
                policy.update_q_entry(states[t], actions[t], new_q_value);
            }
        }
    }
    Ok(())
}

/// tells if the pair state, action at time t are the first visit in the episode
///
/// ## Parameters
///
/// `state`: identifier of the state
/// `action`: identifier of the action
/// `states`: vector of episode's states
/// `actions`: vector of episode's actions
/// `t`: timestamp of the episode
///
/// ## Returns
///
/// true if the pair s,a is the first time it appear in the episode
fn is_first_visit(state: i32, action: i32, states: &[i32], actions: &[i32], t: usize) -> bool {
    for i in 0..t {
        if states[i] == state && actions[i] == action {
            return false;
        }
    }
    true
}