pub mod cliff_walking;
pub mod frozen;
pub mod simple_maze;
pub mod terror_maze;
pub mod windy_gridworld;

/// Tabular environment add methods to Tabular specific case.
///
/// For instance, number of actions and states that in the continuous
/// case are not meaningful.
pub trait TabularEnvironment {
    /// Returns the number of states
    fn get_number_states(&self) -> i32;

    /// Returns the number of actions
    fn get_number_actions(&self) -> i32;
}
