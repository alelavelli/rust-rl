//! Learn module
//!
//! The module contains the implementations of learning algorithms

use std::{error::Error, fmt::Debug};

use ndarray::{ArrayBase, Dim, OwnedRepr};
use rlenv::EnvironmentError;

use crate::{policy::PolicyError, EpisodeError};

pub mod model_free;
pub mod planning;

#[derive(thiserror::Error, Debug)]
pub enum LearningError<S, A> {
    #[error("Failed to make policy step from state {state}. Got error {source}")]
    PolicyStep { state: S, source: PolicyError<A> },

    #[error("Failed to make environment step with action {action}. Got error {source}")]
    EnvironmentStep {
        source: EnvironmentError<S, A>,
        action: A,
    },

    #[error("Failed to generate episode")]
    EpisodeGeneration(#[source] EpisodeError<S, A>),

    #[error("Failed to use model")]
    ModelError,

    #[error("Invalid parameters")]
    InvalidParametersError,

    #[error("Unknown Error when generating episode. Got error {0}")]
    Unknown(#[source] Box<dyn Error>),
}

impl
    From<
        LearningError<
            ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
            ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
        >,
    > for LearningError<ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>, i32>
{
    fn from(
        value: LearningError<
            ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
            ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
        >,
    ) -> Self {
        value.into()
    }
}

/// This struct contains parameters for learning algorithms that define
/// verbosity configurations. According to them different level of progress
/// will be shown to the console
pub struct VerbosityConfig {
    /// true to render the environment during learning
    pub render_env: bool,
    /// true to render learning progress
    pub learning_progress: bool,
    /// true to show progress bar at episode level
    pub episode_progress: bool,
}

pub type ContinuousLearningError = LearningError<ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>, i32>;
