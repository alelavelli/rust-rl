//! Regressor module contains traits and structs for machine learning regressors
//! that are use as base blocks by other components in the library.

use std::error::Error;

use ndarray::{Array2, ArrayBase, Dim, ViewRepr};

use crate::preprocessing::Preprocessor;

pub mod linear;

#[derive(thiserror::Error, Debug)]
pub enum RegressorError {
    #[error("")]
    FitError { source: Box<dyn Error> },
}

pub trait Regressor {
    /// fit the model with the dataset
    fn fit(
        &mut self,
        input: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
        output: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<(), RegressorError>;

    /// predict the target from the input
    fn predict(&mut self, input: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>) -> Array2<f32>;
}

pub struct RegressionPipeline<T: Regressor> {
    input_processing: Vec<Box<dyn Preprocessor<f32>>>,
    output_processing: Vec<Box<dyn Preprocessor<f32>>>,
    regressor: T,
}

impl<T> RegressionPipeline<T>
where
    T: Regressor,
{
    pub fn new(
        input_processing: Vec<Box<dyn Preprocessor<f32>>>,
        output_processing: Vec<Box<dyn Preprocessor<f32>>>,
        regressor: T,
    ) -> RegressionPipeline<T> {
        RegressionPipeline {
            input_processing,
            output_processing,
            regressor,
        }
    }
}

impl<T> Regressor for RegressionPipeline<T>
where
    T: Regressor,
{
    fn fit(
        &mut self,
        input: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
        output: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<(), RegressorError> {
        // fit and transform the input
        let mut current_input = input.to_owned();
        for input_proc in self.input_processing.iter_mut() {
            input_proc.fit(&current_input.view()).unwrap();
            current_input = input_proc.transform(&current_input.view()).unwrap();
        }
        // fit and transform the output
        let mut current_output = output.to_owned();
        for output_proc in self.output_processing.iter_mut() {
            output_proc.fit(&current_output.view()).unwrap();
            current_output = output_proc.transform(&current_output.view()).unwrap();
        }
        // fit the regressor
        self.regressor
            .fit(&current_input.view(), &current_output.view())
            .unwrap();
        Ok(())
    }

    fn predict(&mut self, input: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>) -> Array2<f32> {
        let mut current_input = input.to_owned();
        for input_proc in self.input_processing.iter_mut() {
            current_input = input_proc.transform(&current_input.view()).unwrap();
        }
        self.regressor.predict(&current_input.view())
    }
}
