//! Deterministic model
//!
//! This struct represents a deterministic model

use std::collections::{HashMap, HashSet};

use ndarray::{Array, Array2};
use rand::{seq::IteratorRandom, Rng};

use crate::{
    model::{Model, ModelStep, SampleSA},
    StateAction,
};

pub struct DeterministicModel {
    // transition matrix provides the next state for state s and action a
    // rows are states and columns are actions
    transition_matrix: Array2<i32>,
    // reward matrix provides the reward taking action a on state s
    // rows are states and columns are actions
    reward_matrix: Array2<f32>,
    // experienced samples. state, action pairs the model has seen
    experienced_samples: HashSet<(i32, i32)>,
    // hashmap providing vector of state action pairs that precede the state
    precedessors: HashMap<i32, Vec<StateAction<i32, i32>>>,
}

impl DeterministicModel {
    pub fn new(number_states: usize, number_actions: usize) -> DeterministicModel {
        DeterministicModel {
            transition_matrix: Array::zeros((number_states, number_actions)),
            reward_matrix: Array::zeros((number_states, number_actions)),
            experienced_samples: HashSet::new(),
            precedessors: HashMap::new(),
        }
    }
}

impl Model for DeterministicModel {
    type State = i32;
    type Action = i32;

    fn predict_step(&self, state: Self::State, action: Self::Action) -> ModelStep<Self::State> {
        ModelStep {
            state: self.transition_matrix[[state as usize, action as usize]],
            reward: self.reward_matrix[[state as usize, action as usize]],
        }
    }

    fn update_step(
        &mut self,
        state: Self::State,
        action: Self::Action,
        next_state: Self::State,
        reward: f32,
    ) {
        self.experienced_samples.insert((state, action));
        self.transition_matrix[[state as usize, action as usize]] = next_state;
        self.reward_matrix[[state as usize, action as usize]] = reward;
        let sa = StateAction { state, action };
        let vector = self.precedessors.entry(next_state).or_insert(Vec::new());
        if !vector.contains(&sa) {
            vector.push(sa)
        }
    }

    fn sample_sa<R>(&self, rng: &mut R) -> Option<SampleSA<Self::State, Self::Action>>
    where
        R: Rng + ?Sized,
    {
        self.experienced_samples
            .iter()
            .choose(rng)
            .map(|sample| SampleSA {
                state: sample.0,
                action: sample.1,
            })
    }

    fn get_preceding_sa(
        &self,
        state: Self::State,
    ) -> Option<&Vec<StateAction<Self::State, Self::Action>>> {
        self.precedessors.get(&state)
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{model::Model, StateAction};

    use super::DeterministicModel;

    #[test]
    fn test_update() {
        let mut model = DeterministicModel::new(5, 2);
        model.update_step(0, 0, 1, 1.0);
        let step = model.predict_step(0, 0);
        assert_eq!(step.reward, 1.0);
        assert_eq!(step.state, 1);
    }

    #[test]
    fn test_sample() {
        let mut rng = StdRng::seed_from_u64(222);
        let mut model = DeterministicModel::new(5, 2);

        // the first time we try to sample we get None because the model never seen anything
        let sample = model.sample_sa(&mut rng);
        assert!(sample.is_none());

        let s0 = 0;
        let a0 = 0;
        let s0_next = 1;
        let r0 = 1.0;
        model.update_step(s0, a0, s0_next, r0);

        // the second time we should see only the sample we inserted
        for _ in 0..5 {
            let sample = model.sample_sa(&mut rng);
            assert!(sample.is_some());
            let sample = sample.unwrap();
            assert_eq!(sample.state, s0);
            assert_eq!(sample.action, a0);
        }

        // we add other samples and we check the sampling is correct
        let s1 = 1;
        let a1 = 1;
        let s1_next = 2;
        let r1 = 2.0;
        model.update_step(s1, a1, s1_next, r1);
        let s2 = 2;
        let a2 = 0;
        let s2_next = 3;
        let r2 = 3.0;
        model.update_step(s2, a2, s2_next, r2);

        let seen_samples = vec![(s0, a0), (s1, a1), (s2, a2)];
        let mut rng = rand::thread_rng();
        for _ in 0..5 {
            let sample = model.sample_sa(&mut rng);
            assert!(sample.is_some());
            let sample = sample.unwrap();
            assert!(seen_samples.contains(&(sample.state, sample.action)));
        }
    }

    #[test]
    fn test_precedessors() {
        let mut rng = StdRng::seed_from_u64(222);
        let mut model = DeterministicModel::new(5, 2);

        // the first time we try to sample we get None because the model never seen anything
        let sample = model.sample_sa(&mut rng);
        assert!(sample.is_none());

        let s0 = 0;
        let a0 = 0;
        let s0_next = 1;
        let r0 = 1.0;
        model.update_step(s0, a0, s0_next, r0);

        // the second time we should see only the sample we inserted
        for _ in 0..5 {
            let sample = model.sample_sa(&mut rng);
            assert!(sample.is_some());
            let sample = sample.unwrap();
            assert_eq!(sample.state, s0);
            assert_eq!(sample.action, a0);
        }

        // we add other samples and we check the sampling is correct
        let s1 = 1;
        let a1 = 1;
        let s1_next = 1;
        let r1 = 2.0;
        model.update_step(s1, a1, s1_next, r1);
        let s2 = 2;
        let a2 = 0;
        let s2_next = 3;
        let r2 = 3.0;
        model.update_step(s2, a2, s2_next, r2);

        assert_eq!(
            model.get_preceding_sa(1).unwrap(),
            &vec![
                StateAction {
                    state: s0,
                    action: a0
                },
                StateAction {
                    state: s1,
                    action: a1
                }
            ]
        )
    }
}
