use ndarray::Array2;

use super::PolicyError;

pub mod egreedy;

/// TabularPolicy that works with tabular environments
pub trait TabularPolicy {
    /// step
    ///
    /// ## Parameters
    ///
    /// `observation`: indentifier of the state
    /// `rng`: random seed
    ///
    /// ## Returns
    ///
    /// `action`: identifier of the action wrapped in Result
    fn step(&self, observation: i32, rng: &mut rand::rngs::ThreadRng) -> Result<i32, PolicyError>;

    /// update q function
    ///
    /// ## Parameters
    ///
    /// `observation`: identifier of the state
    /// `action`: identifier of the action
    /// `value`: value of Q(s, a)
    fn update_q_entry(&mut self, observation: i32, action: i32, value: f32);

    /// set q function
    ///
    /// ## Parameters
    ///
    /// `q`: q matrix
    fn set_q(&mut self, q: Array2<f32>);
}
