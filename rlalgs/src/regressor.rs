//! Regressor module contains traits and structs for machine learning regressors
//! that are use as base blocks by other components in the library.

use std::error::Error;

use ndarray::{concatenate, Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};

use crate::{
    preprocessing::Preprocessor,
    value_function::{DifferentiableStateActionValueFunction, StateActionValueFunction},
};

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
    fn predict(&self, input: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>) -> Array2<f32>;
}

/// Decoration of a regressor that process input and output before using the regressor
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

    fn predict(&self, input: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>) -> Array2<f32> {
        // process input
        let mut current_input = input.to_owned();
        for input_proc in self.input_processing.iter() {
            current_input = input_proc.transform(&current_input.view()).unwrap();
        }
        // make prediction
        let mut current_output = self.regressor.predict(&current_input.view());
        // transform output
        for output_proc in self.output_processing.iter() {
            current_output = output_proc
                .inverse_transform(&current_output.view())
                .unwrap();
        }
        current_output
    }
}

impl<T> StateActionValueFunction for RegressionPipeline<T>
where
    T: Regressor + StateActionValueFunction<Update = Array1<f32>>,
{
    type Update = Array1<f32>;

    fn value(
        &self,
        state: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
    ) -> f32 {
        let mut current_input = concatenate(Axis(0), &[state.view(), action.view()])
            .unwrap()
            .insert_axis(Axis(0));
        for input_proc in self.input_processing.iter() {
            current_input = input_proc.transform(&current_input.view()).unwrap();
        }
        let mut current_output = self.regressor.predict(&current_input.view());
        // transform output
        for output_proc in self.output_processing.iter() {
            current_output = output_proc
                .inverse_transform(&current_output.view())
                .unwrap();
        }
        // we know that there is only one element
        assert_eq!(current_output.shape(), [1, 1]);
        current_output[[0, 0]]
    }

    fn value_batch(
        &self,
        states: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
        actions: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Array1<f32> {
        let mut current_input = concatenate(Axis(1), &[states.view(), actions.view()]).unwrap();
        for input_proc in self.input_processing.iter() {
            current_input = input_proc.transform(&current_input.view()).unwrap();
        }
        let mut current_output = self.regressor.predict(&current_input.view());
        // transform output
        for output_proc in self.output_processing.iter() {
            current_output = output_proc
                .inverse_transform(&current_output.view())
                .unwrap();
        }
        current_output.remove_axis(Axis(1))
    }

    fn update(
        &mut self,
        state: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        update: &Self::Update,
    ) -> Result<(), crate::value_function::ValueFunctionError> {
        let mut current_input = concatenate(Axis(0), &[state.view(), action.view()])
            .unwrap()
            .insert_axis(Axis(0));
        for input_proc in self.input_processing.iter() {
            current_input = input_proc.transform(&current_input.view()).unwrap();
        }
        self.regressor.update(
            &current_input.remove_axis(Axis(0)).view(),
            &Array1::zeros(0).view(),
            update,
        )?;
        Ok(())
    }

    fn reset(&mut self) {
        self.regressor.reset();
    }
}

impl<T> DifferentiableStateActionValueFunction for RegressionPipeline<T>
where
    T: DifferentiableStateActionValueFunction
        + Regressor
        + StateActionValueFunction<Update = Array1<f32>>,
{
    fn gradient(
        &self,
        state: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
    ) -> Array1<f32> {
        let mut current_input = concatenate(Axis(0), &[state.view(), action.view()])
            .unwrap()
            .insert_axis(Axis(0));
        for input_proc in self.input_processing.iter() {
            current_input = input_proc.transform(&current_input.view()).unwrap();
        }
        self.regressor.gradient(
            &current_input.remove_axis(Axis(0)).view(),
            &Array1::zeros(0).view(),
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use rand_distr::Uniform;

    use crate::preprocessing::{normalization::ZScore, polynomial::Polynomial, Preprocessor};

    use super::{linear::LinearRegression, RegressionPipeline, Regressor};

    struct Data {
        x: Array2<f32>,
        y: Array2<f32>,
        _w: Array2<f32>,
    }

    fn init_data(n_samples: usize) -> Data {
        let x = ndarray::Array::random((n_samples, 2), Uniform::new(-5., 5.));
        let w = ndarray::Array::from_shape_vec((2, 1), vec![0.5, 0.1]).unwrap();
        let y = x.dot(&w);
        Data { x, y, _w: w }
    }

    #[test]
    fn test_pipeline() {
        let data = init_data(10);
        let input_processing: Vec<Box<dyn Preprocessor<f32>>> = vec![
            Box::new(ZScore::new()),
            Box::new(Polynomial::new(2, false, 1)),
        ];

        let output_processing: Vec<Box<dyn Preprocessor<f32>>> = vec![Box::new(ZScore::new())];
        let regressor = LinearRegression::default();
        let mut pipeline = RegressionPipeline::new(input_processing, output_processing, regressor);
        pipeline.fit(&data.x.view(), &data.y.view()).unwrap();
        let prediction = pipeline.predict(&data.x.view());
        assert_abs_diff_eq!(data.y, prediction, epsilon = 1e-3);
    }
}
