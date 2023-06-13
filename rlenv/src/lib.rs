//! Rust RL Environment
//! 
//! The library defines envorinment used by RL Agents

use ndarray::{Array, IxDyn};

pub trait Environment {
    /// step
    /// 
    /// ## Parameters
    /// 
    /// `action`: n-dimensional array representing the action to take
    /// 
    /// ## Returns
    /// 
    /// `observation`: n-dimensional array representing the environment observation
    fn step(action: Array<f32, IxDyn>) -> Array<f32, IxDyn>;
}