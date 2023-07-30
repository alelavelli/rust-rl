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
pub trait Environment {
    type State;
    type Action;

    /// Initialize the environment providing the starting state
    fn reset(&mut self) -> Self::State;

    /// Returns if the given state is terminal or not
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Returns the terminal states
    fn get_terminal_states(&self) -> Vec<Self::State>;

    /// step
    ///
    /// ## Parameters
    ///
    /// `action`: identifier of the action
    ///
    /// ## Returns
    ///
    /// `observation`: identifier of the state
    fn step<R>(
        &mut self,
        action: &Self::Action,
        rng: &mut R,
    ) -> Result<Step<Self::State, Self::Action>, EnvironmentError>
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

/// Evironment Essay
///
/// This struct contains knowlegde of an environment that can be used to obtain information
/// without accessing the environment itself.
///
/// For instance the environment essay knows if a state is a terminal one or what are the
/// available action for a given state.
pub trait EnvironmentEssay {
    type State;
    type Action;

    /// Returns if the state is terminal or not
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Returns the reward of the step
    fn compute_reward(&self, state: &Self::State, action: &Self::Action, next_state: &Self::State) -> f32;
}

/// Expand the previous trait adding functions for the discrete action case
///
/// Indeed, the list of available actions for a state is only possilbe for the
/// discrete case.
pub trait DiscreteActionEnvironmentEssay {
    type State;
    type Action;

    /// Returns the available actions in the state
    fn available_actions(&self, state: &Self::State) -> Vec<Self::Action>;
}

#[cfg(feature = "gymnasium")]
pub fn hello() {
    println!("hello");
}

pub fn hello_free() {
    println!("hello free");
}
