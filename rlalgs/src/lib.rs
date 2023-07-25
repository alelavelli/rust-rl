//! Rust RL Algorithms
//!
//! The library contains the main Reinforcement Learning algorithms divided by standard tassonomy
use policy::PolicyError;
use rlenv::EnvironmentError;
use std::{error::Error, fmt::Debug};

pub mod learn;
pub mod model;
pub mod policy;
pub mod utils;

#[derive(thiserror::Error)]
pub enum EpisodeGenerationError {
    #[error("Failed to make policy step")]
    PolicyStep(#[source] PolicyError),

    #[error("Failed to make environment step")]
    EnvironmentStep(#[source] EnvironmentError),

    #[error("Failed to learn")]
    GenericError,
}

impl Debug for EpisodeGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self)?;
        if let Some(source) = self.source() {
            writeln!(f, "Caused by:\n\t{}", source)?;
        }
        Ok(())
    }
}

/// Support struct representing state-action pair
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub struct StateAction<S, A> {
    state: S,
    action: A,
}

use std::borrow::BorrowMut;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::Rng;
use rlenv::{Environment, Episode};

use crate::policy::Policy;

/// generate episode using policy and environment
///
/// ## Parameters
///
/// `policy`: policy to use in the environment
/// `environment`: environment to use
/// `episode_max_len`: maximum number of steps in the episode
/// `rng`: random seed
pub fn generate_episode<P, E, R, S, A>(
    policy: &mut P,
    environment: &mut E,
    episode_max_len: Option<i32>,
    rng: &mut R,
    render_env: bool,
    progress_bar: Option<&MultiProgress>,
) -> Result<Episode<S, A>, EpisodeGenerationError>
where
    P: Policy<S, A>,
    E: Environment<S, A>,
    R: Rng + ?Sized,
    S: Copy,
    A: Copy,
{
    // trace variables
    let mut states: Vec<S> = Vec::new();
    let mut actions: Vec<A> = Vec::new();
    let mut rewards: Vec<f32> = Vec::new();
    // get the initial state
    let mut state = environment.reset();
    let mut action: A;
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
        state = episode_step.next_state;
        reward = episode_step.reward;

        // record r_{t+1}
        rewards.push(reward);

        if render_env {
            environment.render();
        }

        if episode_step.terminated | (step_number as f64 >= episode_max_len) {
            states.push(episode_step.next_state);
            break;
        }
    }
    if let Some(spinner) = opt_spinner.borrow_mut() {
        spinner.finish_with_message("episode done");
    }
    Ok(Episode {
        states,
        actions,
        rewards,
    })
}
