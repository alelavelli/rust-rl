use crate::PolicyError;
use ndarray::{s, Array, Array2};
use ndarray_rand::rand_distr::{Distribution, Uniform, WeightedAliasIndex};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;

use super::TabularPolicy;

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

    use crate::policy::tabular::egreedy::EGreedyTabularPolicy;
    use crate::policy::tabular::TabularPolicy;

    #[test]
    fn deterministic_greedy_policy_step() {
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
    fn egreede_policy_step() {
        let n_states = 2;
        let n_actions = 3;
        let epsilon = 0.8;
        let mut pi = EGreedyTabularPolicy::new(n_states, n_actions, epsilon);
        pi.q = Array::zeros((n_states, n_actions));
        // state 0 - best action 0
        pi.q[[0, 0]] = 1.0;
        // state 1 - best action 2
        pi.q[[1, 2]] = 1.0;

        // we make many steps for state 0 and state 1 recording the number of times a action it taken
        // the empirical frequency will be epsilon / n_actions for non optimal and epsilon / n_actions + 1 - epsilon for the optimal
        let best_prob = 1.0 - epsilon + epsilon / n_actions as f32;
        let other_prob = epsilon / n_actions as f32;

        let mut rng = rand::thread_rng();
        let mut occurrencies: Vec<i32> = vec![0; n_actions];
        let n_samples = 1000;
        for _ in 0..n_samples {
            occurrencies[pi.step(0, &mut rng).unwrap() as usize] += 1;
        }
        let probs: Vec<f32> = occurrencies
            .iter()
            .map(|x| *x as f32 / n_samples as f32)
            .collect();

        let tol = 0.1;
        println!(" probs {:?}", probs);
        println!("best {}", best_prob);
        println!("other {}", other_prob);
        assert!((probs[0] - best_prob).abs() < tol);
        assert!((probs[1] - other_prob).abs() < tol);
        assert!((probs[2] - other_prob).abs() < tol);
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