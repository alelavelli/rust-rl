//! Regressor module contains traits and structs for machine learning regressors
//! that are use as base blocks by other components in the library.

use std::error::Error;

use ndarray::Array;

#[derive(thiserror::Error, Debug)]
pub enum RegressorError {
    #[error("")]
    FitError { source: Box<dyn Error> },
}

pub trait Regressor<I, O> {
    type Input;
    type Output;

    /// fit the model with the dataset
    fn fit(&mut self, input: Array<&Self::Input, I>, output: Array<&Self::Input, O>) -> &mut Self;

    /// predict the target from the input
    fn predict(&mut self, input: Array<&Self::Input, I>) -> Array<&Self::Input, O>;
}
