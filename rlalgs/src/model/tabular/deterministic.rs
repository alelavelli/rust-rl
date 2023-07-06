//! Deterministic model
//!
//! This struct represents a deterministic model

use std::collections::HashSet;

use ndarray::{Array, Array2};
use rand::{seq::IteratorRandom, Rng};

use super::{TabularModel, TabularModelStep, TabularSampleSA};

pub struct DeterministicModel {
    // transition matrix provides the next state for state s and action a
    // rows are states and columns are actions
    transition_matrix: Array2<i32>,
    // reward matrix provides the reward taking action a on state s
    // rows are states and columns are actions
    reward_matrix: Array2<f32>,
    // experienced samples. state, action pairs the model has seen
    experienced_samples: HashSet<(i32, i32)>
}

impl DeterministicModel {
    pub fn new(
        number_states: usize,
        number_actions: usize,
        zero_reward: bool,
    ) -> DeterministicModel {
        DeterministicModel {
            transition_matrix: Array::zeros((number_states, number_actions)),
            reward_matrix: if zero_reward {
                Array::zeros((number_states, number_actions))
            } else {
                Array::zeros((number_states, number_actions))
            },
            experienced_samples: HashSet::new()
        }
    }
}

impl TabularModel for DeterministicModel {
    fn predict_step(&self, state: i32, action: i32) -> TabularModelStep {
        TabularModelStep {
            state: self.transition_matrix[[state as usize, action as usize]],
            reward: self.reward_matrix[[state as usize, action as usize]],
        }
    }

    fn update_step(&mut self, state: i32, action: i32, next_state: i32, reward: f32) {
        self.experienced_samples.insert((state, action));
        self.transition_matrix[[state as usize, action as usize]] = next_state;
        self.reward_matrix[[state as usize, action as usize]] = reward;
    }

    fn sample_sa<R>(&self, rng: &mut R) -> Option<TabularSampleSA>
    where
        R: Rng + ?Sized,
    {
        if let Some(sample) = self.experienced_samples.iter().choose(rng) {
            Some(TabularSampleSA { state: sample.0, action: sample.1 })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::tabular::TabularModel;

    use super::DeterministicModel;

    #[test]
    fn test_update() {
        let mut model = DeterministicModel::new(5, 2, true);
        model.update_step(0, 0, 1, 1.0);
        let step = model.predict_step(0, 0);
        assert_eq!(step.reward, 1.0);
        assert_eq!(step.state, 1);
    }
}
