use crate::{PolicyError, TabularPolicy};
use ndarray::{s, Array, Array2};
use ndarray_rand::rand_distr::{Distribution, Uniform, WeightedAliasIndex};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;

/// EGreedyTabularPolicy
///
/// this policy stores the state-value function of the environment and chooses action
/// that maximize it
pub struct EGreedyTabularPolicy {
    // first dimension are the states and the second dimensio are the actions
    pub q: Array2<f32>,
    pub epsilon: f32,
}

impl EGreedyTabularPolicy {
    /// Create new epsilon-greedy Policy initializing the state action value function with random values
    pub fn new(number_state: usize, number_actions: usize, epsilon: f32) -> EGreedyTabularPolicy {
        EGreedyTabularPolicy {
            q: Array::random((number_state, number_actions), Uniform::new(-1., 1.)),
            epsilon,
        }
    }
}

impl TabularPolicy for EGreedyTabularPolicy {
    fn step(&self, observation: i32, rng: &mut rand::rngs::ThreadRng) -> Result<i32, PolicyError> {
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
        Ok(pi.sample(rng) as i32)
    }

    fn set_q(&mut self, q: Array2<f32>) {
        self.q = q
    }

    fn update_q_entry(&mut self, observation: i32, action: i32, value: f32) {
        self.q[[observation as usize, action as usize]] = value;
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array;

    use crate::{policy::EGreedyTabularPolicy, TabularPolicy};

    #[test]
    fn greedy_policy_step() {
        let n_states = 2;
        let n_actions = 5;
        let mut pi = EGreedyTabularPolicy::new(n_states, n_actions, 0.0);
        pi.q = Array::zeros((n_states, n_actions));
        pi.q[[0, 0]] = 10.0;
        pi.q[[1, 1]] = 10.0;
        let mut rng = rand::thread_rng();
        assert_eq!(pi.step(0, &mut rng).unwrap(), 0);
        assert_eq!(pi.step(1, &mut rng).unwrap(), 1);
    }

    #[test]
    fn update_q_entry() {
        let n_states = 2;
        let n_actions = 5;
        let mut pi = EGreedyTabularPolicy::new(n_states, n_actions, 0.0);
        pi.update_q_entry(0, 0, 5.0);
        assert_eq!(pi.q[[0, 0]], 5.0);
    }
}
