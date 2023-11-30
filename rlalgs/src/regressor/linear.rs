//! Linear implementation of value function approximator
use ndarray::{concatenate, Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};
use ndarray_linalg::LeastSquaresSvd;
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use crate::value_function::StateActionValueFunction;

use super::{Regressor, RegressorError};

/// Linear Regression model
pub struct LinearRegression {
    dim: Option<usize>,
    pub weights: Option<Array2<f32>>,
    step_size: f32,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            dim: None,
            weights: None,
            step_size: 1e-3,
        }
    }
}

impl LinearRegression {
    pub fn new(dim: Option<usize>, step_size: f32) -> LinearRegression {
        let weights = if let Some(actual_dim) = dim {
            Some(LinearRegression::init_weights(actual_dim))
        } else {
            None
        };
        LinearRegression {
            dim,
            weights,
            step_size,
        }
    }

    /// Inizialize weight vector randomly between -1 and 1
    fn init_weights(dim: usize) -> ndarray::Array2<f32> {
        ndarray::Array::random((dim, 1), Uniform::new(-1., 1.))
    }
}

/// Implementation of StateActionValueFunction for continuous state and action
impl StateActionValueFunction for LinearRegression {
    fn value(
        &self,
        state: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
    ) -> f32 {
        if let Some(weights) = &self.weights {
            let input = concatenate(Axis(0), &[state.view(), action.view()]).unwrap();
            input.dot(weights)[0]
        } else {
            // use Result as returning type
            panic!("weights are missing")
        }
    }

    fn value_batch(
        &self,
        states: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
        actions: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Array1<f32> {
        if let Some(weights) = &self.weights {
            let input = concatenate(Axis(1), &[states.view(), actions.view()]).unwrap();
            let result = input.dot(weights);
            result.into_iter().collect()
        } else {
            panic!("weights are missing")
        }
    }

    fn update(
        &mut self,
        state: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        action: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
        observed_return: f32,
    ) -> Result<(), crate::value_function::ValueFunctionError> {
        let input = concatenate(Axis(0), &[state.view(), action.view()]).unwrap();
        let weights = if let Some(weights) = &self.weights {
            weights.clone()
        } else {
            self.dim = Some(input.shape()[1]);
            LinearRegression::init_weights(input.shape()[1])
        };

        let delta = observed_return - self.value(state, action);
        let update = delta * input;
        self.weights = Some(weights + self.step_size * update.insert_axis(Axis(1)));
        Ok(())
    }

    fn reset(&mut self) {
        if let Some(dim) = self.dim {
            self.weights = Some(LinearRegression::init_weights(dim))
        } else {
            panic!("missing dim");
        }
    }
}

impl Regressor for LinearRegression {
    fn fit(
        &mut self,
        input: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
        output: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<(), RegressorError> {
        self.dim = Some(input.shape()[1]);
        self.weights = Some(input.least_squares(output).unwrap().solution);
        Ok(())
    }

    fn predict(&mut self, input: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>) -> Array2<f32> {
        if let Some(weights) = &self.weights {
            input.dot(weights)
        } else {
            panic!("missing weights");
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use ndarray_rand::RandomExt;
    use rand_distr::Uniform;

    use crate::{regressor::Regressor, value_function::StateActionValueFunction};

    use super::LinearRegression;

    use approx::assert_abs_diff_eq;

    struct Data {
        x: Array2<f32>,
        y: Array2<f32>,
        w: Array2<f32>,
    }

    fn init_data(n_samples: usize) -> Data {
        let x = ndarray::Array::random((n_samples, 2), Uniform::new(-1., 1.));
        let w = ndarray::Array::from_shape_vec((2, 1), vec![0.5, 0.1]).unwrap();
        let y = x.dot(&w);
        Data { x, y, w }
    }

    #[test]
    fn test_regressor_trait() {
        let data = init_data(100);
        let mut linreg = LinearRegression::new(Some(data.x.shape()[1]), 0.05);
        linreg.fit(&data.x.view(), &data.y.view()).unwrap();
        assert_abs_diff_eq!(data.y, linreg.predict(&data.x.view()), epsilon = 1e-3);
        assert_abs_diff_eq!(data.w, linreg.weights.unwrap(), epsilon = 1e-3);
    }

    #[test]
    fn test_sa_value_function_trait() {
        let data = init_data(100);
        let mut linreg = LinearRegression::new(Some(data.x.shape()[1]), 0.05);

        for _ in 0..5 {
            for i in 0..(data.x.shape()[0]) {
                let sample_state = Array1::from_shape_vec(1, vec![data.x.column(0)[i]]).unwrap();
                let sample_action = Array1::from_shape_vec(1, vec![data.x.column(1)[i]]).unwrap();
                let sample_target = data.y.column(0)[i];
                linreg
                    .update(&sample_state.view(), &sample_action.view(), sample_target)
                    .unwrap();
            }
        }
        // verificare le shape dei pesi
        assert_abs_diff_eq!(data.w, linreg.weights.unwrap(), epsilon = 1e-3);
    }
}
