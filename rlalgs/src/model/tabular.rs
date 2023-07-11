//! Tabular model module

use rand::Rng;

use crate::TabularStateAction;

pub mod deterministic;

/// Struct returned by the model
pub struct TabularModelStep {
    pub state: i32,
    pub reward: f32,
}

/// Struct representing sample state, action
pub struct TabularSampleSA {
    pub state: i32,
    pub action: i32,
}

/// Model trait
///
/// defines methods a model must have
pub trait TabularModel {
    /// Predicts the next state and reward given a state, action pair
    ///
    /// ## Parameters
    ///
    /// - `state`: identifier of the state
    /// - `action`: identifier of the action
    fn predict_step(&self, state: i32, action: i32) -> TabularModelStep;

    /// update the model giving the step information
    ///
    /// ## Parameters
    ///
    /// - `state`: identifier of the state
    /// - `action`: identifier of the action
    /// - `next_state`: resulting state taking the action in the state
    /// - `reward`: obtained reward taking the action in the state
    fn update_step(&mut self, state: i32, action: i32, next_state: i32, reward: f32);

    /// Sample a state, action pair the model has previously experienced
    fn sample_sa<R>(&self, rng: &mut R) -> Option<TabularSampleSA>
    where
        R: Rng + ?Sized;

    /// Returns states that precede the given state
    fn get_preceding_sa(&self, state: i32) -> Option<&Vec<TabularStateAction>>;
}
