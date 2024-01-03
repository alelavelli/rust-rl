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
pub enum CliffWalkingStateType {
    Start,
    Normal,
    Goal,
    Cliff,
}

impl fmt::Display for CliffWalkingStateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CliffWalkingStateType::Normal => "•",
                CliffWalkingStateType::Cliff => "V",
                CliffWalkingStateType::Start => "S",
                CliffWalkingStateType::Goal => "G",
            }
        )
    }
}

/// Cliff walking gridworld tabular environment
///
/// Cliff walking is a standard gridworld with start and goal states and with a region
/// called "the cliff" that gives a reward of -100 and sends the agent instantly back
/// to the start
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
///   - Reach goal (G): 0
///   - the cliff: (C): -100
///   - Any other step: -1
pub struct CliffWalking {
    map_dim: (i32, i32),
    initial_row: i32,
    initial_col: i32,
    n_actions: i32,
    map: Array2<CliffWalkingStateType>,
    current_row: i32,
    current_col: i32,
}

impl CliffWalking {
    pub fn new() -> CliffWalking {
        let initial_row = 3;
        let initial_col = 0;
        CliffWalking {
            map_dim: (4, 10),
            initial_row,
            initial_col,
            n_actions: 4,
            map: array![
                [
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                ],
                [
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                ],
                [
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                    CliffWalkingStateType::Normal,
                ],
                [
                    CliffWalkingStateType::Start,
                    CliffWalkingStateType::Cliff,
                    CliffWalkingStateType::Cliff,
                    CliffWalkingStateType::Cliff,
                    CliffWalkingStateType::Cliff,
                    CliffWalkingStateType::Cliff,
                    CliffWalkingStateType::Cliff,
                    CliffWalkingStateType::Cliff,
                    CliffWalkingStateType::Cliff,
                    CliffWalkingStateType::Goal,
                ],
            ],
            current_row: initial_row,
            current_col: initial_col,
        }
    }

    fn get_state_id(&self, row: &i32, col: &i32) -> i32 {
        row * self.map_dim.1 + col
    }

    fn get_state_type(&self, state: &i32) -> &CliffWalkingStateType {
        let (row, col) = self.to_row_col(state);
        &self.map[[row as usize, col as usize]]
    }

    fn get_state_reward(&self, state: &i32) -> f32 {
        match self.get_state_type(state) {
            CliffWalkingStateType::Goal => 0.0,
            CliffWalkingStateType::Cliff => -100.0,
            _ => -1.0,
        }
    }

    fn to_row_col(&self, state: &i32) -> (i32, i32) {
        let row = (*state as f32 / self.map_dim.1 as f32).floor() as i32;
        let col = state - row * self.map_dim.1;
        (row, col)
    }
}

impl Default for CliffWalking {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for CliffWalking {
    type State = i32;
    type Action = i32;

    fn reset<R>(&mut self, _rng: &mut R) -> Self::State
    where
        R: rand::Rng + ?Sized,
    {
        self.current_row = self.initial_row;
        self.current_col = self.initial_col;
        self.get_state_id(&self.current_row, &self.current_col)
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        let (row, col) = self.to_row_col(state);
        matches!(
            self.map[[row as usize, col as usize]],
            CliffWalkingStateType::Goal
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

        let mut new_row = self.current_row;
        let mut new_col = self.current_col;
        if !Environment::is_terminal(
            self,
            &self.get_state_id(&self.current_row, &self.current_col),
        ) {
            match *action {
                LEFT => new_col = cmp::max(new_col - 1, 0),
                DOWN => new_row = cmp::min(new_row + 1, self.map_dim.0 - 1),
                RIGHT => new_col = cmp::min(new_col + 1, self.map_dim.1 - 1),
                UP => new_row = cmp::max(new_row - 1, 0),
                _ => return Err(EnvironmentError::InvalidAction(*action)),
            };

            self.current_row = new_row;
            self.current_col = new_col;

            // If we reach the cliff then the next state is the start state
            if matches!(
                self.map[[new_row as usize, new_col as usize]],
                CliffWalkingStateType::Cliff
            ) {
                self.current_row = self.initial_row;
                self.current_col = self.initial_col;
            }
        }

        Ok(Step {
            state: starting_state,
            action: *action,
            next_state: self.get_state_id(&self.current_row, &self.current_col),
            // here we use new_row and new_col because in case of cliff the reward is -100 but the current state is start
            reward: self.get_state_reward(&self.get_state_id(&new_row, &new_col)),
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

impl TabularEnvironment for CliffWalking {
    fn get_number_states(&self) -> i32 {
        self.map_dim.0 * self.map_dim.1
    }

    fn get_number_actions(&self) -> i32 {
        self.n_actions
    }

    fn get_terminal_states(&self) -> Vec<i32> {
        let mut terminal_states = Vec::<i32>::new();
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

impl EnvironmentEssay for CliffWalking {
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

impl DiscreteActionEnvironmentEssay for CliffWalking {
    type State = i32;

    type Action = i32;

    fn available_actions(&self, _state: &Self::State) -> Vec<Self::Action> {
        vec![LEFT, RIGHT, UP, DOWN]
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tabular::cliff_walking::{CliffWalkingStateType, DOWN, LEFT, RIGHT, UP},
        Environment,
    };

    use super::CliffWalking;

    #[test]
    fn test_reset() {
        let env = CliffWalking::new();

        assert_eq!(env.current_col, 0);
        assert_eq!(env.current_row, 3);
        assert_eq!(env.get_state_id(&env.current_row, &env.current_col), 30);
        assert_eq!(
            env.get_state_reward(&env.get_state_id(&env.current_row, &env.current_col)),
            -1.0
        );
        assert_eq!(
            *env.get_state_type(&env.get_state_id(&env.current_row, &env.current_col)),
            CliffWalkingStateType::Start
        );
    }

    #[test]
    fn test_step_left() {
        let mut env = CliffWalking::new();
        let mut rng = rand::thread_rng();

        let step = env.step(&LEFT, &mut rng).unwrap();

        assert_eq!(step.next_state, 30);
        assert_eq!(step.reward, -1.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_right_cliff() {
        let mut env = CliffWalking::new();
        let mut rng = rand::thread_rng();

        let step = env.step(&RIGHT, &mut rng).unwrap();

        assert_eq!(step.next_state, 30);
        assert_eq!(step.reward, -100.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_down() {
        let mut env = CliffWalking::new();
        let mut rng = rand::thread_rng();

        let step = env.step(&DOWN, &mut rng).unwrap();

        assert_eq!(step.next_state, 30);
        assert_eq!(step.reward, -1.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_up() {
        let mut env = CliffWalking::new();
        let mut rng = rand::thread_rng();

        let step = env.step(&UP, &mut rng).unwrap();

        assert_eq!(step.next_state, 20);
        assert_eq!(step.reward, -1.0);
        assert_eq!(step.terminated, false);
        assert_eq!(step.truncated, false);
    }

    #[test]
    fn test_step_goal() {
        let mut env = CliffWalking::new();
        env.current_row = 2;
        env.current_col = 9;
        let mut rng = rand::thread_rng();

        let step = env.step(&DOWN, &mut rng).unwrap();

        assert_eq!(step.next_state, 39);
        assert_eq!(step.reward, 0.0);
        assert_eq!(step.terminated, true);
        assert_eq!(step.truncated, false);
        assert_eq!(
            *env.get_state_type(&env.get_state_id(&env.current_row, &env.current_col)),
            CliffWalkingStateType::Goal
        );
    }
}
