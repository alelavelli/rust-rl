//! Model modules
//!
//! The module contains the implementations of algorithms that
//! learn the environment dynamics

use rand::Rng;

use crate::StateAction;

pub mod tabular;

/// Struct returned by the model
pub struct ModelStep<S> {
    pub state: S,
    pub reward: f32,
}

/// Struct representing sample state, action
pub struct SampleSA<S, A> {
    pub state: S,
    pub action: A,
}

/// Model trait
///
/// defines methods a model must have
pub trait Model {
    type State;
    type Action;

    /// Predicts the next state and reward given a state, action pair
    ///
    /// ## Parameters
    ///
    /// - `state`: identifier of the state
    /// - `action`: identifier of the action
    fn predict_step(&self, state: &Self::State, action: &Self::Action) -> ModelStep<Self::State>;

    /// update the model giving the step information
    ///
    /// ## Parameters
    ///
    /// - `state`: starting state
    /// - `action`: taken action
    /// - `next_state`: resulting state taking the action in the state
    /// - `reward`: obtained reward taking the action in the state
    fn update_step(
        &mut self,
        state: &Self::State,
        action: &Self::Action,
        next_state: &Self::State,
        reward: f32,
    );

    /// Sample a state, action pair the model has previously experienced
    fn sample_sa<R>(&self, rng: &mut R) -> Option<SampleSA<Self::State, Self::Action>>
    where
        R: Rng + ?Sized;

    /// Returns states that precede the given state
    fn get_preceding_sa(
        &self,
        state: &Self::State,
    ) -> Option<&Vec<StateAction<Self::State, Self::Action>>>;
}
