pub mod mountain_car;

/// Continuous environment
///
/// Trait for continuous environments in both states and in actions.
pub trait ContinuousEnvironment {
    /// returns the state space
    fn get_state_space(&self) -> ContinuousDataSpace;

    /// returns the action space
    fn get_action_space(&self) -> ContinuousDataSpace;
}

/// Discrete action environment
///
/// Trait for continuous environments in states and with discrete actions.
pub trait DiscreteActionContinuousEnvironment {
    /// returns the state space
    fn get_state_space(&self) -> ContinuousDataSpace;

    /// Returns the number of actions
    fn get_number_actions(&self) -> i32;
}

pub struct DataRange<T> {
    pub low: T,
    pub high: T,
}

/// Data space for continuous environment
///
///
pub struct ContinuousDataSpace {
    /// dimensions of the space that expresses only the number of features
    pub dimensions: u32,
    /// Data range for each dimension
    pub range: Vec<DataRange<f32>>,
}
