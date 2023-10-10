//! Linear implementation of value function approximator

use ndarray::{IxDyn, ArrayD};

use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use super::StateActionValueFunction;

fn build_array(state: &Vec<f64>, action: &Vec<f64>) -> ArrayD<f64> {
    ndarray::ArrayD::from_shape_vec(
        IxDyn(&[1, state.len() + action.len()]),
        [&state[..], &action[..]].concat()
    ).unwrap()
}

/// Value function approximator as linear regression
pub struct LinearVF {
    dim: usize,
    weights: ndarray::Array<f64, IxDyn>,
    step_size: f64,
}

impl LinearVF {
    pub fn new(dim: usize, step_size: f64) -> LinearVF {
        LinearVF {
            dim,
            weights: LinearVF::init_weights(dim),
            step_size,
        }
    }

    /// Inizialize weight vector randomly between -1 and 1
    fn init_weights(dim: usize) -> ndarray::Array<f64, IxDyn> {
        ndarray::ArrayD::random(
            IxDyn(&[1, dim]), 
            Uniform::new(-1., 1.)
        )
    }
}

impl StateActionValueFunction for LinearVF {
    type State = Vec<f64>;
    type Action = Vec<f64>;

    fn value(&self, state: &Self::State, action: &Self::Action) -> f32 {
        let input = build_array(state, action);
        input.dot(self.weights)
    }

    fn value_batch(&self, states: Vec<&Self::State>, actions: Vec<&Self::Action>) -> Vec<f32> {
        todo!()
    }

    fn update(
        &mut self,
        state: &Self::State,
        action: &Self::Action,
        observed_return: f32,
    ) -> Result<(), super::ValueFunctionError> {
        todo!()
    }

    fn update_batch(
        &mut self,
        states: Vec<&Self::State>,
        actions: Vec<&Self::Action>,
        returns: Vec<f32>,
    ) -> Result<(), super::ValueFunctionError> {
        todo!()
    }

    fn reset(&mut self) {
        self.weights = LinearVF::init_weights(self.dim)
    }
   
}