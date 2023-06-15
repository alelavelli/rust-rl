use ndarray::{Array, Array2, s};
use ndarray_rand::rand::{Rng, self};
use ndarray_rand::rand_distr::{Uniform, WeightedAliasIndex, Distribution};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;

use crate::{TabularPolicy, PolicyError};


/// StateValuePolicy
///
/// this policy stores the state-value function of the environment and chooses action
/// that maximize it
pub struct EGreedyTabularPolicy {
    // first dimension are the states and the second dimensio are the actions
    q: Array2<f32>,
    epsilon: f32,
    rng: rand::rngs::ThreadRng
}

impl EGreedyTabularPolicy {
    /// Create new epsilon-greedy Policy initializing the state action value function with random values
    pub fn new(
        number_state: usize,
        number_actions: usize,
        epsilon: f32,
        rng: rand::rngs::ThreadRng
    ) -> EGreedyTabularPolicy {
        EGreedyTabularPolicy {
            q: Array::random((number_state, number_actions), Uniform::new(-1., 1.)),
            epsilon,
            rng
        }
    }
}

impl TabularPolicy for EGreedyTabularPolicy {
    fn step(&mut self, observation: i32) -> Result<i32, PolicyError> {
        // The operations are the followings:
        //   1- get the Q(s, a) values for the given state observation
        //   2- find the action with maximum value
        //   3- create vector of probabilities
        //   4- sample from the distribution
        let q_values = self.q.slice(s![observation, ..]);
        let optimal_action: usize = q_values.argmax().map_err(|_| PolicyError::GenericError)?;
        let num_actions = self.q.shape()[1];
        let mut probabilities: Vec<f32> = vec![self.epsilon / num_actions as f32; num_actions];
        probabilities[optimal_action] += 1.0 - self.epsilon;
        let pi = WeightedAliasIndex::new(probabilities).map_err(|_| PolicyError::GenericError)?;
        Ok(pi.sample(&mut self.rng) as i32)
    }
}