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

/// Support struct representing state-action pair for tabular case
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub struct TabularStateAction {
    state: i32,
    action: i32,
}
