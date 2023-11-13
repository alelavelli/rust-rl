use crate::policy::{Policy, PolicyError, ValuePolicy};
use crate::preprocessing::OneHotEncoder;
use crate::regressor::linear::LinearRegression;
use crate::value_function::vf_enum::ValueFunctionEnum;
use crate::value_function::StateActionValueFunction;
use ndarray::{s, Array, Array1, Array2};
use ndarray_rand::rand_distr::{Uniform, WeightedAliasIndex};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use rand::seq::IteratorRandom;
use rand::Rng;
use rand_distr::WeightedError;

/// Epsilon greedy policy
///
/// The policy chooses the best action with probability $1 - \epsilon$ and the other actions
/// with probability $\epsilon$.
/// This policy works with discrete action space.
/// It is a value-based policy that needs the state-action value function, it can be
/// a table in case of discrete state space or a function approximator otherwise.
///
/// The struct has generic type for Q function that is different according to the action space.
/// The fields `state_dim` and `action_dim` define the state and action space. E-greedy policy
/// is defined for discrete action space only, therefore, `action_dim` represents the number
/// of discrete actions. Whereas, the state space can be either discrete or continuous, when
/// it is discrete it represents the number of states and when it is continuous it represents
/// the number of dimensions that describe the state.
pub struct EGreedyPolicy<Q> {
    q: Q,
    epsilon: f32,
    state_dim: usize,
    action_dim: usize,
}

impl<Q> Clone for EGreedyPolicy<Q>
where
    Q: Clone,
{
    fn clone(&self) -> Self {
        Self {
            q: self.q.clone(),
            epsilon: self.epsilon,
            state_dim: self.state_dim,
            action_dim: self.action_dim,
        }
    }
}

impl<Q> EGreedyPolicy<Q> {
    fn optimal_action<R>(&self, q_values: &Array1<f32>, rng: &mut R) -> i32
    where
        R: Rng + ?Sized,
    {
        // if there are multiple best actions then we take one of them randomly
        let max_value = q_values.max().unwrap();
        let mut best_actions: Vec<usize> = Vec::new();
        let n_actions = q_values.len();
        for i in 0..n_actions {
            if q_values[i] == *max_value {
                best_actions.push(i);
            }
        }
        *best_actions.iter().choose(rng).unwrap() as i32
    }

    fn actions_probabilities<R>(&self, q_values: &Array1<f32>, rng: &mut R) -> Array1<f32>
    where
        R: Rng + ?Sized,
    {
        let optimal_action = self.optimal_action(q_values, rng);
        let mut probabilities: Array1<f32> =
            Array1::from_elem(self.action_dim, self.epsilon / self.action_dim as f32);
        probabilities[optimal_action as usize] += 1.0 - self.epsilon;
        probabilities
    }

    fn sample_action<R>(&self, q_values: &Array1<f32>, rng: &mut R) -> Result<i32, WeightedError>
    where
        R: Rng + ?Sized,
    {
        let probabilities = self.actions_probabilities(q_values, rng).to_vec();
        let pi = WeightedAliasIndex::new(probabilities)?;
        Ok(rand_distr::Distribution::sample(&pi, rng) as i32)
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }
}

impl EGreedyPolicy<Array2<f32>> {
    /// Creates a discrete state e-greedy policy that has the state-action
    /// value function as two dimensional array that stores for each state-action
    /// entry the estimated value.
    ///
    /// Parameters
    ///
    /// - `number_states`: number of states in the environment
    /// - `number_actions`: number of actions in the environment
    /// - `epsilon`: exploration factor
    /// - `zero_q`: if initialize q table with zeros
    pub fn new_discrete(
        number_states: usize,
        number_actions: usize,
        epsilon: f32,
        zero_q: bool,
    ) -> EGreedyPolicy<Array2<f32>> {
        EGreedyPolicy {
            q: if zero_q {
                Array::zeros((number_states, number_actions))
            } else {
                Array::random((number_states, number_actions), Uniform::new(-1., 1.))
            },
            epsilon,
            state_dim: number_states,
            action_dim: number_actions,
        }
    }

    fn action_values_for_state(&self, state: &i32) -> Array1<f32> {
        self.q.slice(s![*state, ..]).to_owned()
    }
}

// action type is Vec<f32> because actions need to be transformed into
// one hot encoding vector before be used by the q function
type ContinuousQ = Box<dyn StateActionValueFunction<f32, f32>>;
impl EGreedyPolicy<ContinuousQ> {
    /// Creates a continuous state and discrete actions e-greedy policy.
    /// The state-action value function is represented by a regressor that takes
    /// as input the state and the action and returns the estimated value
    pub fn new_continuous(
        state_dim: usize,
        number_actions: usize,
        epsilon: f32,
        q_approximator: ValueFunctionEnum,
    ) -> EGreedyPolicy<ContinuousQ> {
        let q = match q_approximator {
            ValueFunctionEnum::LinearRegression { step_size } => {
                // the input dimension is equal to state_dim + number_actions because we transform
                // the discrete action column into one hot encoding
                LinearRegression::new(state_dim + number_actions, step_size)
            }
        };
        EGreedyPolicy {
            q: Box::new(q),
            epsilon,
            state_dim,
            action_dim: number_actions,
        }
    }

    fn action_values_for_state(&self, state: &Array1<f32>) -> Array1<f32> {
        // replicate the state as many times are the actions
        let states = Array::from_shape_fn((self.action_dim, state.shape()[0]), |(_, j)| state[j]);
        let encoder = OneHotEncoder::new(self.action_dim);
        let actions = encoder.transform(
            &Array1::range(0.0, self.action_dim as f32, 1.0)
                .map(|x| *x as i32)
                .view(),
        );
        self.q.value_batch(&states.view(), &actions.view())
    }
}

impl Policy for EGreedyPolicy<Array2<f32>> {
    type State = i32;
    type Action = i32;

    fn step<R>(
        &self,
        state: &Self::State,
        rng: &mut R,
    ) -> Result<Self::Action, PolicyError<Self::Action>>
    where
        R: Rng + ?Sized,
    {
        let q_values = self.action_values_for_state(state);
        Ok(self
            .sample_action(&q_values, rng)
            .map_err(|err| PolicyError::GenericError {
                state,
                source: Box::new(err),
            })
            .unwrap() as Self::Action)
    }

    fn get_best_a(&self, state: &Self::State) -> Result<Self::Action, PolicyError<Self::Action>> {
        Ok(self.optimal_action(
            &self.action_values_for_state(state),
            &mut rand::thread_rng(),
        ) as Self::Action)
    }

    fn action_prob(&self, state: &Self::State, action: &Self::Action) -> f32 {
        let probabilities = self.actions_probabilities(
            &self.action_values_for_state(state),
            &mut rand::thread_rng(),
        );
        probabilities[*action as usize]
    }
}

impl ValuePolicy for EGreedyPolicy<Array2<f32>> {
    type State = i32;
    type Action = i32;
    type Q = Array2<f32>;

    fn get_q(&self) -> &Array2<f32> {
        &self.q
    }

    fn set_q(&mut self, q: Array2<f32>) {
        self.q = q
    }

    fn update_q_entry(&mut self, state: &Self::State, action: &Self::Action, value: f32) {
        self.q[[*state as usize, *action as usize]] = value;
    }

    fn get_q_value(&self, state: &Self::State, action: &Self::Action) -> f32 {
        self.q[[*state as usize, *action as usize]]
    }

    fn get_max_q_value(&self, state: &Self::State) -> Result<f32, PolicyError<Self::State>> {
        let q_values = Array1::from(self.action_values_for_state(state));
        q_values
            .max()
            .map_err(|err| PolicyError::<Self::State>::GenericError {
                state: *state,
                source: Box::new(err),
            })
            .copied()
    }

    fn expected_q_value(&self, state: &Self::State) -> f32 {
        let probabilities = Array1::from(self.actions_probabilities(
            &self.action_values_for_state(state),
            &mut rand::thread_rng(),
        ));
        let q_values = Array1::from(self.action_values_for_state(state));
        q_values.dot(&probabilities)
    }
}

impl Policy for EGreedyPolicy<ContinuousQ> {
    type State = Array1<f32>;
    type Action = i32;

    fn step<R>(
        &self,
        state: &Self::State,
        rng: &mut R,
    ) -> Result<Self::Action, PolicyError<Self::State>>
    where
        R: Rng + ?Sized,
    {
        let q_values = self.action_values_for_state(state);
        Ok(self
            .sample_action(&q_values, rng)
            .map_err(|err| PolicyError::GenericError {
                state,
                source: Box::new(err),
            })
            .unwrap() as Self::Action)
    }

    fn get_best_a(&self, state: &Self::State) -> Result<Self::Action, PolicyError<Self::Action>> {
        Ok(self.optimal_action(
            &self.action_values_for_state(state),
            &mut rand::thread_rng(),
        ) as Self::Action)
    }

    fn action_prob(&self, state: &Self::State, action: &Self::Action) -> f32 {
        let probabilities = self.actions_probabilities(
            &self.action_values_for_state(state),
            &mut rand::thread_rng(),
        );
        probabilities[*action as usize]
    }
}

impl ValuePolicy for EGreedyPolicy<ContinuousQ> {
    type State = Array1<f32>;

    type Action = i32;

    type Q = ContinuousQ;

    fn set_q(&mut self, q: Self::Q) {
        self.q = q;
    }

    fn get_q(&self) -> &Self::Q {
        &self.q
    }

    fn update_q_entry(&mut self, state: &Self::State, action: &Self::Action, value: f32) {
        let encoder = OneHotEncoder::new(self.action_dim);
        self.q
            .update(&state.view(), &encoder.transform_elem(action).view(), value)
            .unwrap();
    }

    fn get_q_value(&self, state: &Self::State, action: &Self::Action) -> f32 {
        let encoder = OneHotEncoder::new(self.action_dim);
        self.q
            .value(&state.view(), &encoder.transform_elem(action).view())
    }

    fn get_max_q_value(&self, state: &Self::State) -> Result<f32, PolicyError<Self::State>> {
        let q_values = Array1::from(self.action_values_for_state(state));
        q_values
            .max()
            .map_err(|err| PolicyError::<Self::State>::GenericError {
                state: state.clone(),
                source: Box::new(err),
            })
            .copied()
    }

    fn expected_q_value(&self, state: &Self::State) -> f32 {
        let probabilities = Array1::from(self.actions_probabilities(
            &self.action_values_for_state(state),
            &mut rand::thread_rng(),
        ));
        let q_values = Array1::from(self.action_values_for_state(state));
        q_values.dot(&probabilities)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, Array1};

    use crate::{
        policy::{egreedy::EGreedyPolicy, Policy, ValuePolicy},
        value_function::vf_enum::ValueFunctionEnum::LinearRegression,
    };

    #[test]
    fn deterministic_greedy_policy_step_discrete() {
        let n_states = 2;
        let n_actions = 5;
        let mut pi = EGreedyPolicy::new_discrete(n_states, n_actions, 0.0, false);
        pi.q = Array::zeros((n_states, n_actions));
        pi.q[[0, 0]] = 10.0;
        pi.q[[1, 1]] = 10.0;
        let mut rng = rand::thread_rng();
        assert_eq!(pi.step(&0, &mut rng).unwrap(), 0);
        assert_eq!(pi.step(&1, &mut rng).unwrap(), 1);
    }

    #[test]
    fn deterministic_greedy_policy_step_continuous() {
        let state_dim = 1;
        let n_actions = 2;
        let mut pi = EGreedyPolicy::new_continuous(
            state_dim,
            n_actions,
            0.0,
            LinearRegression { step_size: 1.0 },
        );
        let state = Array1::zeros(1);
        let action = 1;
        pi.update_q_entry(&state, &action, 100.0);
        let mut rng = rand::thread_rng();
        assert_eq!(pi.step(&state, &mut rng).unwrap(), action);
    }

    #[test]
    fn egreedy_policy_step_discrete() {
        let n_states = 2;
        let n_actions = 3;
        let epsilon = 0.8;
        let mut pi = EGreedyPolicy::new_discrete(n_states, n_actions, epsilon, false);
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
            occurrencies[pi.step(&0, &mut rng).unwrap() as usize] += 1;
        }
        let probs: Vec<f32> = occurrencies
            .iter()
            .map(|x| *x as f32 / n_samples as f32)
            .collect();

        let tol = 0.1;

        assert!((probs[0] - best_prob).abs() < tol);
        assert!((probs[1] - other_prob).abs() < tol);
        assert!((probs[2] - other_prob).abs() < tol);
    }

    #[test]
    fn egreedy_policy_step_continuous() {
        let epsilon = 0.8;
        let state_dim = 1;
        let n_actions = 2;
        let mut pi = EGreedyPolicy::new_continuous(
            state_dim,
            n_actions,
            epsilon,
            LinearRegression { step_size: 1.0 },
        );
        let state = Array1::zeros(1);
        let action = 0;
        pi.update_q_entry(&state, &action, 100.0);

        // we make many steps for state 0 and state 1 recording the number of times a action it taken
        // the empirical frequency will be epsilon / n_actions for non optimal and epsilon / n_actions + 1 - epsilon for the optimal
        let best_prob = 1.0 - epsilon + epsilon / n_actions as f32;
        let other_prob = epsilon / n_actions as f32;

        let mut rng = rand::thread_rng();
        let mut occurrencies: Vec<i32> = vec![0; n_actions];
        let n_samples = 1000;
        for _ in 0..n_samples {
            occurrencies[pi.step(&state, &mut rng).unwrap() as usize] += 1;
        }
        let probs: Vec<f32> = occurrencies
            .iter()
            .map(|x| *x as f32 / n_samples as f32)
            .collect();

        let tol = 0.1;

        assert!((probs[0] - best_prob).abs() < tol);
        assert!((probs[1] - other_prob).abs() < tol);
    }

    #[test]
    fn update_q_entry() {
        let n_states = 2;
        let n_actions = 5;
        let mut pi = EGreedyPolicy::new_discrete(n_states, n_actions, 0.0, false);
        pi.update_q_entry(&0, &0, 5.0);
        assert_eq!(pi.q[[0, 0]], 5.0);
    }
}
