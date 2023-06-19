use crate::EnvironmentError;

pub mod frozen;

/// Trait for Tabular Environments
///
/// It defines basic interafces a tabular environment must have.
///  
/// ## Methods
///
/// `reset`: reset the environment to an initial state
/// `is_terminal`: wheter the state is terminal or not
/// `get_number_states`: number of states
/// `get_number_actions`: number of actions
/// `step`: step of the environment taking agent action
pub trait TabularEnvironment {
    /// Initialize the environment providing the starting state
    fn reset(&mut self) -> i32;

    /// Returns if the given state is terminal or not
    fn is_terminal(&self, state: i32) -> bool;

    /// Returns the number of states
    fn get_number_states(&self) -> i32;

    /// Returns the number of actions
    fn get_number_actions(&self) -> i32;

    /// step
    ///
    /// ## Parameters
    ///
    /// `action`: identifier of the action
    ///
    /// ## Returns
    ///
    /// `observation`: identifier of the state
    fn step(
        &mut self,
        action: i32,
        rng: &mut rand::rngs::ThreadRng,
    ) -> Result<TabularStep, EnvironmentError>;
}

/// TabularEpisode
///
/// Struct containing the information about an episode the agent
/// did with the environment
pub struct TabularEpisode {
    pub states: Vec<i32>,
    pub actions: Vec<i32>,
    pub rewards: Vec<f32>,
}

/// TabularStep
///
/// This struct is returned after a step in the environment
///
/// ## Fields
///
/// `observation`: identifier of the state
/// `reward`: reward got taking the action
/// `terminated`: whether a terminal state of the MDP is reached
/// `truncated`: whether a truncation condition outside the scope of the MDP is satisfied.
/// Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
/// Can be used to end the episode prematurely before a `terminal state` is reached.

pub struct TabularStep {
    pub observation: i32,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}