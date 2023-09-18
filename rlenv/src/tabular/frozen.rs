use std::{cmp, fmt};

use ndarray::{array, Array2};
use rand::Rng;

use crate::{
    DiscreteActionEnvironmentEssay, Environment, EnvironmentError, EnvironmentEssay, Step,
};
use colored::Colorize;

use super::TabularEnvironment;

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

impl fmt::Display for FrozenLakeStateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                FrozenLakeStateType::Frozen => "•",
                FrozenLakeStateType::Hole => "X",
                FrozenLakeStateType::Start => "S",
                FrozenLakeStateType::Goal => "G",
            }
        )
    }
}

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
/// where both row and col start from 0
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

    fn get_state_id(&self, row: &i32, col: &i32) -> i32 {
        row * self.map_dim.1 + col
    }

    fn get_state_type(&self, state: &i32) -> &FrozenLakeStateType {
        let (row, col) = self.to_row_col(state);
        &self.map[[row as usize, col as usize]]
    }

    fn get_state_reward(&self, state: &i32) -> f32 {
        match self.get_state_type(state) {
            FrozenLakeStateType::Goal => 1.0,
            _ => 0.0,
        }
    }

    fn to_row_col(&self, state: &i32) -> (i32, i32) {
        let row = (*state as f32 / self.map_dim.1 as f32).floor() as i32;
        let col = state - row * self.map_dim.1;
        (row, col)
    }
}

impl Default for FrozenLake {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for FrozenLake {
    type State = i32;
    type Action = i32;

    fn reset(&mut self) -> Self::State {
        self.current_row = self.initial_row;
        self.current_col = self.initial_col;
        self.get_state_id(&self.current_row, &self.current_col)
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        let (row, col) = self.to_row_col(state);
        matches!(
            self.map[[row as usize, col as usize]],
            FrozenLakeStateType::Hole | FrozenLakeStateType::Goal
        )
    }

    fn step<R>(
        &mut self,
        action: &Self::Action,
        #[allow(unused_variables)] rng: &mut R,
    ) -> Result<Step<Self::State, Self::Action>, crate::EnvironmentError<Self::State, Self::Action>>
    where
        R: Rng + ?Sized,
    {
        let starting_state = self.get_state_id(&self.current_row, &self.current_col);

        if !Environment::is_terminal(
            self,
            &self.get_state_id(&self.current_row, &self.current_col),
        ) {
            let mut new_row = self.current_row;
            let mut new_col = self.current_col;

            match *action {
                LEFT => new_col = cmp::max(self.current_col - 1, 0),
                DOWN => new_row = cmp::min(self.current_row + 1, self.map_dim.0 - 1),
                RIGHT => new_col = cmp::min(self.current_col + 1, self.map_dim.1 - 1),
                UP => new_row = cmp::max(self.current_row - 1, 0),
                _ => return Err(EnvironmentError::InvalidAction(*action)),
            };

            self.current_row = new_row;
            self.current_col = new_col;
        }

        Ok(Step {
            state: starting_state,
            action: *action,
            next_state: self.get_state_id(&self.current_row, &self.current_col),
            reward: self.get_state_reward(&self.get_state_id(&self.current_row, &self.current_col)),
            terminated: Environment::is_terminal(
                self,
                &self.get_state_id(&self.current_row, &self.current_col),
            ),
            truncated: false,
        })
    }

    fn render(&self) {
        println!("----");
        for row in 0..self.map_dim.0 {
            for col in 0..self.map_dim.1 {
                if (row == self.current_row) & (col == self.current_col) {
                    print!(
                        "{}",
                        &format!("{:^5}", self.map[[row as usize, col as usize]]).on_color("red")
                    )
                } else {
                    print!(
                        "{}",
                        &format!("{:^5}", self.map[[row as usize, col as usize]])
                    )
                }
            }
            println!();
        }
        println!("----");
    }

    fn set_state(&mut self, state: &Self::State) {
        (self.current_row, self.current_col) = self.to_row_col(state);
    }
}

impl TabularEnvironment for FrozenLake {
    type State = i32;

    fn get_number_states(&self) -> i32 {
        self.map_dim.0 * self.map_dim.1
    }

    fn get_number_actions(&self) -> i32 {
        self.n_actions
    }

    fn get_terminal_states(&self) -> Vec<Self::State> {
        let mut terminal_states = Vec::<Self::State>::new();
        for row in 0..self.map_dim.0 {
            for col in 0..self.map_dim.1 {
                let state = self.get_state_id(&row, &col);
                if Environment::is_terminal(self, &state) {
                    terminal_states.push(state);
                }
            }
        }
        terminal_states
    }
}

impl EnvironmentEssay for FrozenLake {
    type State = i32;
    type Action = i32;

    fn is_terminal(&self, state: &Self::State) -> bool {
        Environment::is_terminal(self, state)
    }

    fn compute_reward(
        &self,
        _state: &Self::State,
        _action: &Self::Action,
        next_state: &Self::State,
    ) -> f32 {
        self.get_state_reward(next_state)
    }
}

impl DiscreteActionEnvironmentEssay for FrozenLake {
    type State = i32;

    type Action = i32;

    fn available_actions(&self, _state: &Self::State) -> Vec<Self::Action> {
        vec![LEFT, RIGHT, UP, DOWN]
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tabular::frozen::{FrozenLakeStateType, DOWN, LEFT, RIGHT, UP},
        Environment,
    };

    use super::FrozenLake;

    #[test]
    fn test_reset() {
        let env = FrozenLake::new();

        assert_eq!(env.current_col, 0);
        assert_eq!(env.current_row, 0);
        assert_eq!(env.get_state_id(&env.current_row, &env.current_col), 0);
        assert_eq!(
            env.get_state_reward(&env.get_state_id(&env.current_row, &env.current_col)),
            0.0
        );
        assert_eq!(
            *env.get_state_type(&env.get_state_id(&env.current_row, &env.current_col)),
            FrozenLakeStateType::Start
        );
    }

    #[test]
    fn test_step_left() {
        let mut env = FrozenLake::new();
        let mut rng = rand::thread_rng();

        let step = env.step(&LEFT, &mut rng).unwrap();

        assert_eq!(step.next_state, 0);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_right() {
        let mut env = FrozenLake::new();
        let mut rng = rand::thread_rng();

        let step = env.step(&RIGHT, &mut rng).unwrap();

        assert_eq!(step.next_state, 1);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_down() {
        let mut env = FrozenLake::new();
        let mut rng = rand::thread_rng();

        let step = env.step(&DOWN, &mut rng).unwrap();

        assert_eq!(step.next_state, 4);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_up() {
        let mut env = FrozenLake::new();
        let mut rng = rand::thread_rng();

        let step = env.step(&UP, &mut rng).unwrap();

        assert_eq!(step.next_state, 0);
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

        let step = env.step(&RIGHT, &mut rng).unwrap();

        assert_eq!(step.next_state, 5);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_eq!(
            *env.get_state_type(&env.get_state_id(&env.current_row, &env.current_col)),
            FrozenLakeStateType::Hole
        );
    }

    #[test]
    fn test_step_goal() {
        let mut env = FrozenLake::new();
        env.current_row = 3;
        env.current_col = 2;
        let mut rng = rand::thread_rng();

        let step = env.step(&RIGHT, &mut rng).unwrap();

        assert_eq!(step.next_state, 15);
        assert_eq!(step.reward, 1.0);
        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_eq!(
            *env.get_state_type(&env.get_state_id(&env.current_row, &env.current_col)),
            FrozenLakeStateType::Goal
        );
    }
}
