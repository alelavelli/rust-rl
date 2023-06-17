//! Rust RL Algorithms
//!
//! The library contains the main Reinforcement Learning algorithms divided by standard tassonomy
use ndarray::Array2;
use ndarray_rand::rand;
use rlenv::EnvironmentError;
use std::{error::Error, fmt::Debug};
use thiserror;

pub mod learn;
pub mod policy;

#[derive(thiserror::Error)]
pub enum PolicyError {
    #[error("Failed to compute action")]
    GenericError,
}

impl Debug for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self)?;
        if let Some(source) = self.source() {
            writeln!(f, "Caused by:\n\t{}", source)?;
        }
        Ok(())
    }
}

/// TabularPolicy that works with tabular environments
pub trait TabularPolicy {
    /// step
    ///
    /// ## Parameters
    ///
    /// `observation`: indentifier of the state
    /// `rng`: random seed
    ///
    /// ## Returns
    ///
    /// `action`: identifier of the action wrapped in Result
    fn step(&self, observation: i32, rng: &mut rand::rngs::ThreadRng) -> Result<i32, PolicyError>;

    /// update q function
    ///
    /// ## Parameters
    ///
    /// `observation`: identifier of the state
    /// `action`: identifier of the action
    /// `value`: value of Q(s, a)
    fn update_q_entry(&mut self, observation: i32, action: i32, value: f32);

    /// set q function
    ///
    /// ## Parameters
    ///
    /// `q`: q matrix
    fn set_q(&mut self, q: Array2<f32>);
}

#[derive(thiserror::Error)]
pub enum LearningError {
    #[error("Failed to generate episode")]
    EpisodeGeneration(#[source] EpisodeGenerationError),

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
