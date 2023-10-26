//! Regressor module contains traits and structs for machine learning regressors
//! that are use as base blocks by other components in the library.

use std::error::Error;

use ndarray::Array2;

pub mod linear;

#[derive(thiserror::Error, Debug)]
pub enum RegressorError {
    #[error("")]
    FitError { source: Box<dyn Error> },
}

pub trait Regressor {
    type Input;
    type Output;

    /// fit the model with the dataset
    fn fit(&mut self, input: &Array2<Self::Input>, output: &Array2<Self::Input>) -> &mut Self;

    /// predict the target from the input
    fn predict(&mut self, input: &Array2<Self::Input>) -> Array2<Self::Output>;
}
