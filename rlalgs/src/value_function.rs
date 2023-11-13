//! Value function module contains structs and traits for State and State-Action Value function
//! in continuous environments.

pub mod vf_enum;

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
pub trait StateActionValueFunction<S, A> {
    /// returns the estimated value of a single pair
    fn value(
        &self,
        state: &ArrayBase<ViewRepr<&S>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&A>, Dim<[usize; 1]>>,
    ) -> f32;

    /// returns the estimated values for an array of pairs
    fn value_batch(
        &self,
        states: &ArrayBase<ViewRepr<&S>, Dim<[usize; 2]>>,
        actions: &ArrayBase<ViewRepr<&A>, Dim<[usize; 2]>>,
    ) -> Array1<f32>;

    /// update the model with a new sample
    fn update(
        &mut self,
        state: &ArrayBase<ViewRepr<&S>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&A>, Dim<[usize; 1]>>,
        observed_return: f32,
    ) -> Result<(), ValueFunctionError>;

    /// update the model with a batch of samples
    fn update_batch(
        &mut self,
        states: &ArrayBase<ViewRepr<&S>, Dim<[usize; 2]>>,
        actions: &ArrayBase<ViewRepr<&A>, Dim<[usize; 2]>>,
        observed_returns: Array1<f32>,
    ) -> Result<(), ValueFunctionError> {
        for i in 0..states.len() {
            self.update(
                &states.index_axis(Axis(0), i),
                &actions.index_axis(Axis(0), i),
                observed_returns[i],
            )
            .unwrap();
        }
        Ok(())
    }

    /// reset the value function
    fn reset(&mut self);
}
