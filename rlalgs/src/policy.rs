//! Policy modules
//!
//! The module contains the implementations of Reinforcement Learning policies

pub mod tabular;
use std::{error::Error, fmt::Debug};

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
