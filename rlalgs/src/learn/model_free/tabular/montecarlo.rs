use std::collections::HashMap;

use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use rand::Rng;
use rlenv::tabular::TabularEnvironment;

use crate::{
    learn::LearningError,
    learn::{model_free::tabular::generate_tabular_episode, VerbosityConfig},
    policy::tabular::TabularPolicy,
};

/// Parameters for montecarlo learning algorithm
///
/// - `episodes`: number of episode to generate during learning
/// - `gamma`: discount factor
/// - `first_visit_mode`: true to apply MonteCarlo First visit
pub struct Params {
    pub episodes: i32,
    pub gamma: f32,
    pub first_visit_mode: bool,
}

/// Monte Carlo on-policy, update rule
/// $$ Q(S_t, A_t) \leftarrow \sum_n^N Returns(S_t, A_t) $$
///
/// ## Parameters
///
/// -`policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `params`: struct with parameter for the learning
/// - `rng`: random generator
/// - `verbosity`: verbosity configs
pub fn learn<P, E, R>(
    mut policy: P,
    mut environment: E,
    params: Params,
    rng: &mut R,
    versbosity: &VerbosityConfig,
) -> Result<P, LearningError>
where
    P: TabularPolicy,
    E: TabularEnvironment,
    R: Rng + ?Sized,
{
    // initialize variables
    let mut returns: HashMap<(i32, i32), Vec<f32>> = HashMap::new();

    let multiprogress_bar = MultiProgress::new();
    let progress_bar = multiprogress_bar.add(ProgressBar::new(params.episodes as u64));

    for _ in (0..params.episodes).progress_with(progress_bar) {
        // GENERATE EPISODE
        let episode = generate_tabular_episode(
            &mut policy,
            &mut environment,
            None,
            rng,
            versbosity.render_env,
            if versbosity.episode_progress {
                Some(&multiprogress_bar)
            } else {
                None
            },
        )
        .map_err(LearningError::EpisodeGeneration)?;
        let (states, actions, rewards) = (episode.states, episode.actions, episode.rewards);
        // Update Q function
        let mut g = 0.0;
        for t in (0..(states.len() - 1)).rev() {
            g = params.gamma * g + rewards[t];
            // If we are in first_visit settings then we check that the pair s,a a time t is the first visit
            // otherwise, we enter always
            if !params.first_visit_mode
                | is_first_visit(states[t], actions[t], &states, &actions, t)
            {
                let sa_returns = returns.entry((states[t], actions[t])).or_insert(Vec::new());
                sa_returns.push(g);
                let new_q_value: f32 = sa_returns.iter().sum::<f32>() / sa_returns.len() as f32;
                policy.update_q_entry(states[t], actions[t], new_q_value);
            }
        }
    }
    Ok(policy)
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

#[cfg(test)]
mod tests {
    use crate::learn::model_free::tabular::montecarlo::is_first_visit;

    #[test]
    fn test_is_first_visit() {
        let states = [3, 3, 3, 1, 4];
        let actions = [6, 5, 6, 5, 6];

        assert!(is_first_visit(3, 6, &states, &actions, 0));
        assert!(!is_first_visit(3, 6, &states, &actions, 1));
        assert!(!is_first_visit(3, 6, &states, &actions, 2));
    }
}
