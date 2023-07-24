use ndarray::Array2;
use rand::Rng;

use super::PolicyError;

pub mod egreedy;
pub mod mcts;

/// TabularPolicy that works with tabular environments
pub trait TabularPolicy {
    /// step
    ///
    /// ## Parameters
    ///
    /// `state`: indentifier of the state
    /// `rng`: random seed
    ///
    /// ## Returns
    ///
    /// `action`: identifier of the action wrapped in Result
    fn step<R>(&self, state: i32, rng: &mut R) -> Result<i32, PolicyError>
    where
        R: Rng + ?Sized;

    /// update q function
    ///
    /// ## Parameters
    ///
    /// `state`: identifier of the state
    /// `action`: identifier of the action
    /// `value`: value of Q(s, a)
    fn update_q_entry(&mut self, state: i32, action: i32, value: f32);

    /// set q function
    ///
    /// ## Parameters
    ///
    /// `q`: q matrix
    fn set_q(&mut self, q: Array2<f32>);

    /// Return q function
    fn get_q(&self) -> &Array2<f32>;

    /// Return q value of state and action
    fn get_q_value(&self, state: i32, action: i32) -> f32;

    /// Return the value of the best action even if it does not represent the policy action
    ///
    /// max_a { Q(S, a) }
    fn get_max_q_value(&self, state: i32) -> Result<f32, PolicyError>;

    /// Return best action
    fn get_best_a(&self, state: i32) -> Result<i32, PolicyError>;

    /// Return the probability to take action in the state
    fn action_prob(&self, state: i32, action: i32) -> f32;

    /// Return expected value for a state
    fn expected_q_value(&self, state: i32) -> f32;
}
