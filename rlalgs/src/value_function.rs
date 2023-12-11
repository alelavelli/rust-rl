//! Value function module contains structs and traits for State and State-Action Value function
//! in continuous environments.

use std::error::Error;

use ndarray::{Array1, ArrayBase, Axis, Dim, ViewRepr};

#[derive(thiserror::Error, Debug)]
pub enum ValueFunctionError {
    #[error("Failed to fit value function model")]
    FitError { source: Box<dyn Error> },
}

/// State-Action value function that from the pair of state and action
/// estimates its value.
///
/// Given a set of s,a pairs and returns then it can learn an estimator
/// to generalize over unseen data.
pub trait StateActionValueFunction {
    type Update;
    /// returns the estimated value of a single pair
    fn value(
        &self,
        state: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
    ) -> f32;

    /// returns the estimated values for an array of pairs
    fn value_batch(
        &self,
        states: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
        actions: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Array1<f32>;

    /// update the model with a new sample
    fn update(
        &mut self,
        state: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        update: &Self::Update,
    ) -> Result<(), ValueFunctionError>;

    /// update the model with a batch of samples
    fn update_batch(
        &mut self,
        states: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
        actions: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
        updates: &[Self::Update],
    ) -> Result<(), ValueFunctionError> {
        for (i, update) in updates.iter().enumerate().take(states.len()) {
            self.update(
                &states.index_axis(Axis(0), i),
                &actions.index_axis(Axis(0), i),
                update,
            )
            .unwrap();
        }
        Ok(())
    }

    /// reset the value function
    fn reset(&mut self);
}

pub trait DifferentiableStateActionValueFunction: StateActionValueFunction {
    fn gradient(
        &self,
        state: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
    ) -> Array1<f32>;
}
