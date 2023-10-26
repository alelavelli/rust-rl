//! Linear implementation of value function approximator
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::LeastSquaresSvd;
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use crate::value_function::StateActionValueFunction;

use super::Regressor;

fn build_array(state: &Vec<f32>, action: &Vec<f32>) -> Array1<f32> {
    ndarray::Array::from_shape_vec(
        [state.len() + action.len()],
        [&state[..], &action[..]].concat(),
    )
    .unwrap()
}

fn build_batch_array(states: &Vec<&Vec<f32>>, actions: &Vec<&Vec<f32>>) -> Array2<f32> {
    // https://docs.rs/ndarray/0.15.6/ndarray/struct.ArrayBase.html#conversions-from-nested-vecsarrays
    let mut arr = Array2::zeros((states.len(), states[0].len() + actions[0].len()));
    for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
        let sa = [&states[i][..], &actions[i][..]].concat();
        for (j, col) in row.iter_mut().enumerate() {
            *col = sa[j];
        }
    }
    arr
}

/// Linear Regression model
pub struct LinearRegression {
    dim: usize,
    pub weights: Array2<f32>,
    step_size: f32,
}

/*
[ ] basis functions
*/

impl LinearRegression {
    pub fn new(dim: usize, step_size: f32) -> LinearRegression {
        LinearRegression {
            dim,
            weights: LinearRegression::init_weights(dim),
            step_size,
        }
    }

    /// Inizialize weight vector randomly between -1 and 1
    fn init_weights(dim: usize) -> ndarray::Array2<f32> {
        ndarray::Array::random((dim, 1), Uniform::new(-1., 1.))
    }
}

impl StateActionValueFunction for LinearRegression {
    type State = Vec<f32>;
    type Action = Vec<f32>;

    fn value(&self, state: &Self::State, action: &Self::Action) -> f32 {
        let input = build_array(state, action);
        input.dot(&self.weights)[0]
    }

    fn value_batch(&self, states: Vec<&Self::State>, actions: Vec<&Self::Action>) -> Vec<f32> {
        let input = build_batch_array(&states, &actions);
        let result = input.dot(&self.weights);
        result.into_iter().collect()
    }

    fn compute_gradient(&self, state: &Self::State, action: &Self::Action) -> Vec<f32> {
        // in linear model the gradient is equal to the input
        build_array(state, action).into_iter().collect()
    }

    fn update_parameters(&mut self, update: Vec<f32>) {
        self.weights = &self.weights
            + self.step_size * Array2::from_shape_vec((update.len(), 1), update.clone()).unwrap();
    }

    fn reset(&mut self) {
        self.weights = LinearRegression::init_weights(self.dim)
    }
}

impl Regressor for LinearRegression {
    type Input = f32;
    type Output = f32;

    fn fit(&mut self, input: &Array2<Self::Input>, output: &Array2<Self::Input>) -> &mut Self {
        self.weights = input.least_squares(&output).unwrap().solution;
        self
    }

    fn predict(&mut self, input: &Array2<Self::Input>) -> Array2<Self::Output> {
        input.dot(&self.weights)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
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
        let mut linreg = LinearRegression::new(data.x.shape()[1], 0.05);
        linreg.fit(&data.x, &data.y);
        assert_abs_diff_eq!(data.w, linreg.weights, epsilon = 1e-3);
        assert_abs_diff_eq!(data.y, linreg.predict(&data.x), epsilon = 1e-3);
    }

    #[test]
    fn test_sa_value_function_trait() {
        let data = init_data(100);
        let mut linreg = LinearRegression::new(data.x.shape()[1], 0.05);
        for _ in 0..5 {
            for i in 0..(data.x.shape()[0]) {
                let sample_state = vec![data.x.column(0)[i]];
                let sample_action = vec![data.x.column(1)[i]];
                let sample_target = data.y.column(0)[i];
                linreg
                    .update(&sample_state, &sample_action, sample_target)
                    .unwrap();
            }
        }
        // verificare le shape dei pesi
        assert_abs_diff_eq!(data.w, linreg.weights, epsilon = 1e-3);
    }
}