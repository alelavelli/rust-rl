//! Rust RL Environment
//!
//! The library defines envorinment used by RL Agents

use std::{error::Error, fmt::Debug};

#[derive(thiserror::Error)]
pub enum EnvironmentError {
    #[error("Failed to make step")]
    GenericError,
}

impl Debug for EnvironmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self)?;
        if let Some(source) = self.source() {
            writeln!(f, "Caused by:\n\t{}", source)?;
        }
        Ok(())
    }
}

pub trait TabularEnvironment {
    fn init(&mut self) -> i32;
    fn is_terminal(&self, state: i32) -> bool;
    /// step
    ///
    /// ## Parameters
    ///
    /// `action`: identifier of the action
    ///
    /// ## Returns
    ///
    /// `observation`: identifier of the state
    fn step(
        &mut self,
        action: i32,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Result<(i32, f32), EnvironmentError>;
}

pub struct TabularEpisode {
    pub states: Vec<i32>,
    pub actions: Vec<i32>,
    pub rewards: Vec<f32>,
}

#[cfg(feature = "gymnasium")]
pub fn hello() {
    println!("hello");
}

pub fn hello_free() {
    println!("hello free");
}
