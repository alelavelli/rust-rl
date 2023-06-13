//! Rust RL Algorithms
//! 
//! The library contains the main Reinforcement Learning algorithms divided by standard tassonomy

use ndarray::{Array, IxDyn};

/// The Policy trait defines interface to interact with the environment as an Agent
pub trait Policy {
    /// step
    /// 
    /// ## Parameters
    /// 
    /// `observation`: n-dimensional array representing the environment observation
    /// 
    /// ## Returns
    /// 
    /// `action`: n-dimensional array representing the action to take
    fn step(observation: Array<f32, IxDyn>) -> Array<f32, IxDyn>;
}