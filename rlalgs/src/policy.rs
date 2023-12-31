//! Policy modules
//!
//! The module contains the implementations of Reinforcement Learning policies

pub mod egreedy;
pub mod tabular;
use std::{error::Error, fmt::Debug};

use ndarray::Array1;
use rand::Rng;

#[derive(thiserror::Error, Debug)]
pub enum PolicyError<S> {
    #[error("Failed to compute action for state {state}")]
    GenericError { state: S, source: Box<dyn Error> },
}

/// Policy trait
///
/// The policy trait has two generic types that represent the State and the Action space
pub trait Policy {
    // State type
    type State;
    // Action type
    type Action;
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
    fn step<R>(
        &self,
        state: &Self::State,
        rng: &mut R,
    ) -> Result<Self::Action, PolicyError<Self::State>>
    where
        R: Rng + ?Sized;

    /// Return best action
    fn get_best_a(&self, state: &Self::State) -> Result<Self::Action, PolicyError<Self::Action>>;

    /// Return the probability to take action in the state
    fn action_prob(&self, state: &Self::State, action: &Self::Action) -> f32;
}

/// Value Policy trait
///
/// Defines methods a value policy must implements
pub trait ValuePolicy {
    type State;
    type Action;
    type Q;
    type Update;

    /// set q function
    ///
    fn set_q(&mut self, q: Self::Q);

    /// Return q function
    fn get_q(&self) -> &Self::Q;

    /// update q function
    ///
    /// ## Parameters
    ///
    /// `state`: state
    /// `action`: action
    /// `value`: value of Q(s, a)
    fn update_q_entry(&mut self, state: &Self::State, action: &Self::Action, value: &Self::Update);

    /// Return q value of state and action
    fn get_q_value(&self, state: &Self::State, action: &Self::Action) -> f32;

    /// Return the value of the best action even if it does not represent the policy action
    ///
    /// max_a { Q(S, a) }
    fn get_max_q_value(&self, state: &Self::State) -> Result<f32, PolicyError<Self::State>>;

    /// Return expected value for a state
    fn expected_q_value(&self, state: &Self::State) -> f32;
}

pub trait DifferentiablePolicy {
    type State;
    type Action;

    fn gradient(&self, state: &Self::State, action: &Self::Action) -> Array1<f32>;
}
