//! Linear implementation of value function approximator

use ndarray::{Array2, Array1, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use super::StateActionValueFunction;

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

/// Value function approximator as linear regression
pub struct LinearVF {
    dim: usize,
    weights: Array2<f32>,
    step_size: f32,
}

/*
[ ] basis functions 
*/

impl LinearVF {
    pub fn new(dim: usize, step_size: f32) -> LinearVF {
        LinearVF {
            dim,
            weights: LinearVF::init_weights(dim),
            step_size,
        }
    }

    /// Inizialize weight vector randomly between -1 and 1
    fn init_weights(dim: usize) -> ndarray::Array2<f32> {
        ndarray::Array::random(
            (1, dim), Uniform::new(-1., 1.)
        )
    }
}

impl StateActionValueFunction for LinearVF {
    type State = Vec<f32>;
    type Action = Vec<f32>;

    fn value(&self, state: &Self::State, action: &Self::Action) -> f32 {
        let input = build_array(state, action);
        self.weights.dot(&input)[0]
    }

    fn value_batch(&self, states: Vec<&Self::State>, actions: Vec<&Self::Action>) -> Vec<f32> {
        let input = build_batch_array(&states, &actions);
        let result = self.weights.dot(&input);
        result.into_iter().collect()
    }

    fn compute_gradient(&self, state: &Self::State, action: &Self::Action) -> Vec<f32> {
        // in linear model the gradient is equal to the input
        build_array(state, action).into_iter().collect()
    }

    fn update_parameters(&mut self, update: Vec<f32>) {
        self.weights = &self.weights + self.step_size * Array1::from(update);
    }

    fn reset(&mut self) {
        self.weights = LinearVF::init_weights(self.dim)
    }
}
