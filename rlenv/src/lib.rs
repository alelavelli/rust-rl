//! Rust RL Environment
//!
//! The library defines envorinment used by RL Agents

use std::{error::Error, fmt::Debug};

pub mod tabular;

#[derive(thiserror::Error)]
pub enum EnvironmentError {
    #[error("Action is not valid")]
    WrongAction,

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

#[cfg(feature = "gymnasium")]
pub fn hello() {
    println!("hello");
}

pub fn hello_free() {
    println!("hello free");
}
