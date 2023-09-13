//! Learn module
//!
//! The module contains the implementations of learning algorithms

use std::{error::Error, fmt::Debug};

use rlenv::EnvironmentError;

use crate::{policy::PolicyError, EpisodeGenerationError};

pub mod model_free;
pub mod planning;

#[derive(thiserror::Error)]
pub enum LearningError {
    #[error("Failed to make policy step")]
    PolicyStep(#[source] PolicyError),

    #[error("Failed to make environment step")]
    EnvironmentStep(#[source] EnvironmentError),

    #[error("Failed to generate episode")]
    EpisodeGeneration(#[source] EpisodeGenerationError),

    #[error("Failed to use model")]
    ModelError,

    #[error("Invalid parameters")]
    InvalidParametersError,

    #[error("Failed to learn")]
    GenericError,
}

impl Debug for LearningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self)?;
        if let Some(source) = self.source() {
            writeln!(f, "Caused by:\n\t{}", source)?;
        }
        Ok(())
    }
}

/// This struct contains parameters for learning algorithms that define
/// verbosity configurations. According to them different level of progress
/// will be shown to the console
pub struct VerbosityConfig {
    // true to render the environment during learning
    pub render_env: bool,
    // true to render learning progress
    pub learning_progress: bool,
    // true to show progress bar at episode level
    pub episode_progress: bool,
}
