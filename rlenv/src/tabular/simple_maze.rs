use std::{cmp, fmt};

use ndarray::{array, Array2};
use rand::Rng;

use crate::{Environment, Step, EnvironmentError};
use colored::Colorize;

use super::TabularEnvironment;

const LEFT: i32 = 0;
const DOWN: i32 = 1;
const RIGHT: i32 = 2;
const UP: i32 = 3;

/// Enumeration for the environment Simple Maze
///
/// Each value contains a bool to indicate if it is a terminal state
#[derive(PartialEq, Debug)]
pub enum SimpleMazeStateType {
    Start,
    Normal,
    Goal,
    Wall,
}

impl fmt::Display for SimpleMazeStateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SimpleMazeStateType::Normal => "•",
                SimpleMazeStateType::Wall => "|",
                SimpleMazeStateType::Start => "S",
                SimpleMazeStateType::Goal => "G",
            }
        )
    }
}

/// Simple Maze gridworld tabular environment
///
/// Simple Maze is a standard gridworld with start and goal states and with a set
/// of obstacles "the walls" that the agent can pass over.
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
///   - Any other step: 0
///
/// The suggested discount factor gamma is 0.95
///
pub struct SimpleMaze {
    map_dim: (i32, i32),
    initial_row: i32,
    initial_col: i32,
    n_actions: i32,
    map: Array2<SimpleMazeStateType>,
    current_row: i32,
    current_col: i32,
}

impl SimpleMaze {
    pub fn new() -> SimpleMaze {
        let initial_row = 2;
        let initial_col = 0;
        SimpleMaze {
            map_dim: (6, 9),
            initial_row,
            initial_col,
            n_actions: 4,
            map: array![
                [
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Wall,
                    SimpleMazeStateType::Goal,
                ],
                [
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Wall,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Wall,
                    SimpleMazeStateType::Normal,
                ],
                [
                    SimpleMazeStateType::Start,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Wall,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Wall,
                    SimpleMazeStateType::Normal,
                ],
                [
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Wall,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                ],
                [
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Wall,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                ],
                [
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                    SimpleMazeStateType::Normal,
                ],
            ],
            current_row: initial_row,
            current_col: initial_col,
        }
    }

    fn get_state_id(&self, row: i32, col: i32) -> i32 {
        row * self.map_dim.1 + col
    }

    fn get_state_type(&self, state: i32) -> &SimpleMazeStateType {
        let (row, col) = self.to_row_col(state);
        &self.map[[row as usize, col as usize]]
    }

    fn get_state_reward(&self, state: i32) -> f32 {
        match self.get_state_type(state) {
            SimpleMazeStateType::Goal => 1.0,
            _ => 0.0,
        }
    }

    fn to_row_col(&self, state: i32) -> (i32, i32) {
        let row = (state as f32 / self.map_dim.1 as f32).floor() as i32;
        let col = state - row * self.map_dim.1;
        (row, col)
    }
}

impl Default for SimpleMaze {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment<i32, i32> for SimpleMaze {
    fn reset(&mut self) -> i32 {
        self.current_row = self.initial_row;
        self.current_col = self.initial_col;
        self.get_state_id(self.current_row, self.current_col)
    }

    fn is_terminal(&self, state: i32) -> bool {
        let (row, col) = self.to_row_col(state);
        matches!(
            self.map[[row as usize, col as usize]],
            SimpleMazeStateType::Goal
        )
    }

    fn get_terminal_states(&self) -> Vec<i32> {
        let mut terminal_states = Vec::<i32>::new();
        for row in 0..self.map_dim.0 {
            for col in 0..self.map_dim.1 {
                let state = self.get_state_id(row, col);
                if self.is_terminal(state) {
                    terminal_states.push(state);
                }
            }
        }
        terminal_states
    }

    fn step<R>(
        &mut self,
        action: i32,
        #[allow(unused_variables)] rng: &mut R,
    ) -> Result<Step<i32, i32>, crate::EnvironmentError>
    where
        R: Rng + ?Sized,
    {
        let starting_state = self.get_state_id(self.current_row, self.current_col);

        let mut new_row = self.current_row;
        let mut new_col = self.current_col;
        if !self.is_terminal(self.get_state_id(self.current_row, self.current_col)) {
            match action {
                LEFT => new_col = cmp::max(new_col - 1, 0),
                DOWN => new_row = cmp::min(new_row + 1, self.map_dim.0 - 1),
                RIGHT => new_col = cmp::min(new_col + 1, self.map_dim.1 - 1),
                UP => new_row = cmp::max(new_row - 1, 0),
                _ => return Err(EnvironmentError::WrongAction),
            };
            // if we encounter a wall then we remain in the same position
            if self.map[[new_row as usize, new_col as usize]] != SimpleMazeStateType::Wall {
                self.current_row = new_row;
                self.current_col = new_col;
            }
        }

        Ok(Step {
            state: starting_state,
            action,
            next_state: self.get_state_id(self.current_row, self.current_col),
            // here we use new_row and new_col because in case of cliff the reward is -100 but the current state is start
            reward: self.get_state_reward(self.get_state_id(self.current_row, self.current_col)),
            terminated: self.is_terminal(self.get_state_id(self.current_row, self.current_col)),
            truncated: false,
        })
    }

    fn render(&self) {
        println!("=========");
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
        println!("=========");
    }
}


impl TabularEnvironment for SimpleMaze {

    fn get_number_states(&self) -> i32 {
        self.map_dim.0 * self.map_dim.1
    }

    fn get_number_actions(&self) -> i32 {
        self.n_actions
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tabular::simple_maze::{SimpleMazeStateType, DOWN, LEFT, RIGHT, UP},
        Environment,
    };

    use super::SimpleMaze;

    #[test]
    fn test_reset() {
        let env = SimpleMaze::new();

        assert_eq!(env.current_col, 0);
        assert_eq!(env.current_row, 2);
        assert_eq!(env.get_state_id(env.current_row, env.current_col), 18);
        assert_eq!(
            env.get_state_reward(env.get_state_id(env.current_row, env.current_col)),
            0.0
        );
        assert_eq!(
            *env.get_state_type(env.get_state_id(env.current_row, env.current_col)),
            SimpleMazeStateType::Start
        );
    }

    #[test]
    fn test_step_left() {
        let mut env = SimpleMaze::new();
        let mut rng = rand::thread_rng();

        let step = env.step(LEFT, &mut rng).unwrap();

        assert_eq!(step.next_state, 18);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_right_wall() {
        let mut env = SimpleMaze::new();
        let mut rng = rand::thread_rng();

        let step = env.step(RIGHT, &mut rng).unwrap();

        assert_eq!(step.next_state, 19);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);

        let step = env.step(RIGHT, &mut rng).unwrap();

        assert_eq!(step.next_state, 19);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_down() {
        let mut env = SimpleMaze::new();
        let mut rng = rand::thread_rng();

        let step = env.step(DOWN, &mut rng).unwrap();

        assert_eq!(step.next_state, 27);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_up() {
        let mut env = SimpleMaze::new();
        let mut rng = rand::thread_rng();

        let step = env.step(UP, &mut rng).unwrap();

        assert_eq!(step.next_state, 9);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_goal() {
        let mut env = SimpleMaze::new();
        env.current_row = 1;
        env.current_col = 8;
        let mut rng = rand::thread_rng();

        let step = env.step(UP, &mut rng).unwrap();

        assert_eq!(step.next_state, 8);
        assert_eq!(step.reward, 1.0);
        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_eq!(
            *env.get_state_type(env.get_state_id(env.current_row, env.current_col)),
            SimpleMazeStateType::Goal
        );
    }
}
