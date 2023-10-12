//! Value function module contains structs and traits for State and State-Action Value function
//! in continuous environments.

use std::error::Error;

use itertools::Itertools;
pub mod linear;

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
    type State;
    type Action;

    /// returns the estimated value of a single pair
    fn value(&self, state: &Self::State, action: &Self::Action) -> f32;

    /// returns the estimated values for an array of pairs
    fn value_batch(&self, states: Vec<&Self::State>, actions: Vec<&Self::Action>) -> Vec<f32>;

    /// compute gradient in the state action
    fn compute_gradient(&self, state: &Self::State, action: &Self::Action) -> Vec<f32>;

    /// update internal parameters
    fn update_parameters(&mut self, update: Vec<f32>);

    /// update the model with a new sample
    fn update(
        &mut self,
        state: &Self::State,
        action: &Self::Action,
        observed_return: f32,
    ) -> Result<(), ValueFunctionError> {
        let gradient = self.compute_gradient(state, action);
        let delta = observed_return - self.value(state, action);
        let update = gradient.iter().map(|x| x * delta).collect_vec();
        self.update_parameters(update);
        Ok(())
    }

    /// update the model with a batch of samples
    fn update_batch(
        &mut self,
        states: Vec<&Self::State>,
        actions: Vec<&Self::Action>,
        observed_returns: Vec<f32>,
    ) -> Result<(), ValueFunctionError> {
        for i in 0..states.len() {
            self.update(states[i], actions[i], observed_returns[i]).unwrap();
        }
        Ok(())
    }

    /// reset the value function
    fn reset(&mut self);
}