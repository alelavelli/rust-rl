//! Rust RL Environment
//!
//! The library defines envorinment used by RL Agents

use std::{error::Error, fmt::Debug};

use rand::Rng;

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

/// Trait for Tabular Environments
///
/// It defines basic interafces an environment must have.
pub trait Environment<S, A> {
    /// Initialize the environment providing the starting state
    fn reset(&mut self) -> S;

    /// Returns if the given state is terminal or not
    fn is_terminal(&self, state: S) -> bool;

    /// Returns the number of states
    fn get_number_states(&self) -> i32;

    /// Returns the number of actions
    fn get_number_actions(&self) -> i32;

    /// Returns the terminal states
    fn get_terminal_states(&self) -> Vec<S>;

    /// step
    ///
    /// ## Parameters
    ///
    /// `action`: identifier of the action
    ///
    /// ## Returns
    ///
    /// `observation`: identifier of the state
    fn step<R>(&mut self, action: A, rng: &mut R) -> Result<Step<S, A>, EnvironmentError>
    where
        R: Rng + ?Sized;

    /// render
    ///
    /// shows the environment to the console
    fn render(&self);
}

/// Episode
///
/// Struct containing the information about an episode the agent
/// did with the environment
#[derive(Debug)]
pub struct Episode<S, A> {
    pub states: Vec<S>,
    pub actions: Vec<A>,
    pub rewards: Vec<f32>,
}

/// Step
///
/// This struct is returned after a step in the environment
///
/// ## Fields
///
/// `state`: starting state
/// `action`: taken action
/// `next_state`: resulting state
/// `reward`: reward got taking the action
/// `terminated`: whether a terminal state of the MDP is reached
/// `truncated`: whether a truncation condition outside the scope of the MDP is satisfied.
/// Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
/// Can be used to end the episode prematurely before a `terminal state` is reached.

pub struct Step<S, A> {
    pub state: S,
    pub action: A,
    pub next_state: S,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}


#[cfg(feature = "gymnasium")]
pub fn hello() {
    println!("hello");
}

pub fn hello_free() {
    println!("hello free");
}
