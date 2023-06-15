//! Rust RL Algorithms
//!
//! The library contains the main Reinforcement Learning algorithms divided by standard tassonomy
use std::{fmt::Debug, error::Error};
use ndarray::{Array, IxDyn};
use thiserror;

pub mod policy;

#[derive(thiserror::Error)]
pub enum PolicyError {
    #[error("Failed to compute action")]
    GenericError
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
    ///
    /// ## Returns
    ///
    /// `action`: identifier of the action
    fn step(&mut self, observation: i32) -> Result<i32, PolicyError>;
}

/// The Policy trait defines interface to interact with the environment as an Agent
pub trait Policy {
    /// step
    ///
    /// ## Parameters
    ///
    /// `observation`: n-dimensional array representing the environment observation
    ///
    /// ## Returns
    ///
    /// `action`: n-dimensional array representing the action to take
    fn step(&mut self, observations: Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>, PolicyError>;
}
