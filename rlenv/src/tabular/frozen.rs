use std::cmp;

use ndarray::{array, Array2};

use crate::{tabular::TabularEnvironment, EnvironmentError};

use super::TabularStep;

/// ForzenLake tabular environment
///
/// Frozen lake involves crossing a frozen lake from an initial state Start (S)
/// to a Goal (G) without falling into any Holes (H) by walking over the Frozen (F) lake.
///
/// ## Action Space
///
/// The agent has four possible actions that represent the direction of the step:
///   - 0: LEFT
///   - 1: DOWN
///   - 2: RIGHT
///   - 3: UP
///
/// ## Observation space
///
/// The observation space is a value representing the agent's current position
/// as a single number. The value is equal to current_row + n_rows + current_col
/// where both row and col start from 0)
///
/// ## Rewards
///
///   - Reach goal (G): +1
///   - Reach hole (H): 0
///   - Reach frozen (F): 0
pub struct FrozenLake {
    map_dim: (i32, i32),
    initial_row: i32,
    initial_col: i32,
    n_actions: i32,
    map: Array2<FrozenLakeStateType>,
    current_row: i32,
    current_col: i32,
}

const LEFT: i32 = 0;
const DOWN: i32 = 1;
const RIGHT: i32 = 2;
const UP: i32 = 3;

/// Enumeration for the environment FrozenLake
///
/// Each value contains a bool to indicate if it is a terminal state
#[derive(PartialEq, Debug)]
pub enum FrozenLakeStateType {
    Start,
    Frozen,
    Hole,
    Goal,
}

impl FrozenLake {
    pub fn new() -> FrozenLake {
        FrozenLake {
            map_dim: (4, 4),
            initial_row: 0,
            initial_col: 0,
            n_actions: 4,
            map: array![
                [
                    FrozenLakeStateType::Start,
                    FrozenLakeStateType::Frozen,
                    FrozenLakeStateType::Frozen,
                    FrozenLakeStateType::Frozen
                ],
                [
                    FrozenLakeStateType::Frozen,
                    FrozenLakeStateType::Hole,
                    FrozenLakeStateType::Frozen,
                    FrozenLakeStateType::Hole
                ],
                [
                    FrozenLakeStateType::Frozen,
                    FrozenLakeStateType::Frozen,
                    FrozenLakeStateType::Frozen,
                    FrozenLakeStateType::Hole
                ],
                [
                    FrozenLakeStateType::Hole,
                    FrozenLakeStateType::Frozen,
                    FrozenLakeStateType::Frozen,
                    FrozenLakeStateType::Goal
                ]
            ],
            current_row: 0,
            current_col: 0,
        }
    }

    fn get_current_state_id(&self) -> i32 {
        self.current_row * self.map_dim.0 + self.current_col
    }

    fn get_current_state_type(&self) -> &FrozenLakeStateType {
        &self.map[[self.current_row as usize, self.current_col as usize]]
    }

    fn to_row_col(&self, state: i32) -> (i32, i32) {
        let row = (state as f32 / self.map_dim.0 as f32).floor() as i32;
        let col = state - row * self.map_dim.1;
        (row, col)
    }

    fn get_current_state_reward(&self) -> f32 {
        match self.map[[self.current_row as usize, self.current_col as usize]] {
            FrozenLakeStateType::Goal => 1.0,
            _ => 0.0,
        }
    }
}

impl Default for FrozenLake {
    fn default() -> Self {
        Self::new()
    }
}

impl TabularEnvironment for FrozenLake {
    fn reset(&mut self) -> i32 {
        self.current_row = self.initial_row;
        self.current_col = self.initial_col;
        self.get_current_state_id()
    }

    fn is_terminal(&self, state: i32) -> bool {
        let (r, c) = self.to_row_col(state);
        matches!(
            self.map[[r as usize, c as usize]],
            FrozenLakeStateType::Hole | FrozenLakeStateType::Goal
        )
    }

    fn step(
        &mut self,
        action: i32,
        #[allow(unused_variables)] rng: &mut rand::rngs::ThreadRng,
    ) -> Result<TabularStep, crate::EnvironmentError> {
        let mut new_row = self.current_row;
        let mut new_col = self.current_col;

        match action {
            LEFT => new_col = cmp::max(self.current_col - 1, 0),
            DOWN => new_row = cmp::min(self.current_row + 1, self.map_dim.0 - 1),
            RIGHT => new_col = cmp::min(self.current_col + 1, self.map_dim.1 - 1),
            UP => new_row = cmp::max(self.current_row - 1, 0),
            _ => return Err(EnvironmentError::WrongAction),
        };

        self.current_row = new_row;
        self.current_col = new_col;

        Ok(TabularStep {
            observation: self.get_current_state_id(),
            reward: self.get_current_state_reward(),
            terminated: self.is_terminal(self.get_current_state_id()),
            truncated: false,
        })
    }

    fn get_number_states(&self) -> i32 {
        self.map_dim.0 * self.map_dim.1
    }

    fn get_number_actions(&self) -> i32 {
        self.n_actions
    }
}

#[cfg(test)]
mod tests {
    use crate::tabular::{
        frozen::{FrozenLakeStateType, DOWN, LEFT, RIGHT, UP},
        TabularEnvironment,
    };

    use super::FrozenLake;

    #[test]
    fn test_reset() {
        let env = FrozenLake::new();

        assert_eq!(env.current_col, 0);
        assert_eq!(env.current_row, 0);
        assert_eq!(env.get_current_state_id(), 0);
        assert_eq!(env.get_current_state_reward(), 0.0);
        assert_eq!(*env.get_current_state_type(), FrozenLakeStateType::Start);
    }

    #[test]
    fn test_step_left() {
        let mut env = FrozenLake::new();
        let mut rng = rand::thread_rng();

        let step = env.step(LEFT, &mut rng).unwrap();

        assert_eq!(step.observation, 0);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_right() {
        let mut env = FrozenLake::new();
        let mut rng = rand::thread_rng();

        let step = env.step(RIGHT, &mut rng).unwrap();

        assert_eq!(step.observation, 1);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_down() {
        let mut env = FrozenLake::new();
        let mut rng = rand::thread_rng();

        let step = env.step(DOWN, &mut rng).unwrap();

        assert_eq!(step.observation, 4);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_up() {
        let mut env = FrozenLake::new();
        let mut rng = rand::thread_rng();

        let step = env.step(UP, &mut rng).unwrap();

        assert_eq!(step.observation, 0);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_hole() {
        let mut env = FrozenLake::new();
        env.current_row = 1;
        env.current_col = 0;
        let mut rng = rand::thread_rng();

        let step = env.step(RIGHT, &mut rng).unwrap();

        assert_eq!(step.observation, 5);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_eq!(*env.get_current_state_type(), FrozenLakeStateType::Hole);
    }

    #[test]
    fn test_step_goal() {
        let mut env = FrozenLake::new();
        env.current_row = 3;
        env.current_col = 2;
        let mut rng = rand::thread_rng();

        let step = env.step(RIGHT, &mut rng).unwrap();

        assert_eq!(step.observation, 15);
        assert_eq!(step.reward, 1.0);
        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_eq!(*env.get_current_state_type(), FrozenLakeStateType::Goal);
    }
}
