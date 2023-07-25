//! Policy modules
//!
//! The module contains the implementations of Reinforcement Learning policies

pub mod tabular;
use std::{error::Error, fmt::Debug};

use rand::Rng;

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

/// Policy trait
///
/// The policy trait has two generic types that represent the State and the Action space
pub trait Policy<S, A> {
    /// step
    ///
    /// ## Parameters
    ///
    /// `state`: state to predict the action
    /// `rng`: random seed
    ///
    /// ## Returns
    ///
    /// `action`: identifier of the action wrapped in Result
    fn step<R>(&self, state: S, rng: &mut R) -> Result<A, PolicyError>
    where
        R: Rng + ?Sized;

    /// Return best action
    fn get_best_a(&self, state: S) -> Result<A, PolicyError>;

    /// Return the probability to take action in the state
    fn action_prob(&self, state: S, action: A) -> f32;
}

/// Value Policy trait
///
/// Defines methods a value policy must implements
pub trait ValuePolicy<S, A, Q> {
    /// set q function
    ///
    fn set_q(&mut self, q: Q);

    /// Return q function
    fn get_q(&self) -> &Q;

    /// update q function
    ///
    /// ## Parameters
    ///
    /// `state`: state
    /// `action`: action
    /// `value`: value of Q(s, a)
    fn update_q_entry(&mut self, state: S, action: A, value: f32);

    /// Return q value of state and action
    fn get_q_value(&self, state: S, action: A) -> f32;

    /// Return the value of the best action even if it does not represent the policy action
    ///
    /// max_a { Q(S, a) }
    fn get_max_q_value(&self, state: S) -> Result<f32, PolicyError>;

    /// Return expected value for a state
    fn expected_q_value(&self, state: S) -> f32;
}
