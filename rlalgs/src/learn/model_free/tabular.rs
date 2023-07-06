pub mod double_qlearning;
pub mod montecarlo;
pub mod n_step_q_sigma;
pub mod n_step_sarsa;
pub mod n_step_tree_backup;
pub mod qlearning;
pub mod sarsa;

use std::borrow::BorrowMut;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
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
    progress_bar: Option<&MultiProgress>,
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

    // If multiprogress bar is provided, then we create the spinner which
    // indicates the progression of the episode
    let mut opt_spinner: Option<ProgressBar> = None;
    if progress_bar.is_some() {
        let mut spinner = ProgressBar::new_spinner();
        spinner.enable_steady_tick(std::time::Duration::from_secs(1));
        spinner.set_style(
            ProgressStyle::default_spinner()
                .tick_strings(&[
                    "▹▹▹▹▹",
                    "▸▹▹▹▹",
                    "▹▸▹▹▹",
                    "▹▹▸▹▹",
                    "▹▹▹▸▹",
                    "▹▹▹▹▸",
                    "▪▪▪▪▪",
                ])
                .template("{spinner:.blue} {msg}")
                .unwrap(),
        );
        spinner = progress_bar.as_ref().unwrap().add(spinner);
        opt_spinner = Some(spinner);
    }
    if let Some(spinner) = opt_spinner.borrow_mut() {
        spinner.set_message("Starting Episode");
    }
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
    if let Some(spinner) = opt_spinner.borrow_mut() {
        spinner.finish_with_message("episode done");
    }
    Ok(TabularEpisode {
        states,
        actions,
        rewards,
    })
}
