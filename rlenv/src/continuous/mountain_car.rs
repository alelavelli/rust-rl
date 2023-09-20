use ndarray::Array2;
use rand_distr::Distribution;

use rand::distributions::Uniform;

use crate::{Environment, Step};

use super::{ContinuousDataSpace, DataRange, DiscreteActionContinuousEnvironment};

const THROTTLE_REVERSE: i32 = 0;
const ZERO: i32 = 1;
const THROTTLE_FORWARD: i32 = 2;

/// Struct containing the two dimensions of the state in the mountain car environment
struct MountainCarState {
    position: f32,
    speed: f32,
}

impl MountainCarState {
    fn to_array(&self) -> Array2<f32> {
        Array2::from(vec![[self.position, self.speed]])
    }

    fn from_array(state: &Array2<f32>) -> MountainCarState {
        MountainCarState {
            position: state[(0, 0)],
            speed: state[(0, 1)],
        }
    }
}

/// MountainCar environment
///
/// This is a continuous environment with discrete action space.
///
/// The state is composed of two variables:
///     - position
///     - speed
///
/// The state evolves with the following dynamics:
/// $$
/// speed_{t+1} = speed_{t} + (action - 1) * force - \cos( 3 * position_t) * gravity
/// $$
/// $$
/// position_{t+1} = position_{t} + speed_{t+1}
/// $$
pub struct MountainCar {
    position_bounds: (f32, f32),
    speed_bounds: (f32, f32),
    force: f32,
    gravity: f32,
    initial_position_range: (f32, f32),
    n_actions: i32,
    current_state: MountainCarState,
}

impl MountainCar {
    pub fn new<R>(rng: &mut R) -> MountainCar
    where
        R: rand::Rng + Sized,
    {
        let mut env = MountainCar {
            position_bounds: (-1.2, 0.6),
            speed_bounds: (-0.7, 0.7),
            force: 0.001,
            gravity: 0.0025,
            initial_position_range: (-0.6, -0.4),
            n_actions: 3,
            current_state: MountainCarState {
                position: -0.4,
                speed: 0.0,
            },
        };
        env.reset(rng);
        env
    }

    fn get_state_reward(&self, state: &MountainCarState) -> f32 {
        if state.position >= 0.5 {
            0.0
        } else {
            -1.0
        }
    }

    fn is_valid_action(&self, action: &i32) -> bool {
        vec![THROTTLE_REVERSE, ZERO, THROTTLE_FORWARD].contains(action)
    }
}

impl Environment for MountainCar {
    type State = Array2<f32>;
    type Action = i32;

    fn reset<R>(&mut self, rng: &mut R) -> Self::State
    where
        R: rand::Rng + ?Sized,
    {
        let sampled_position =
            Uniform::new(self.initial_position_range.0, self.initial_position_range.1).sample(rng);
        self.current_state = MountainCarState {
            position: sampled_position,
            speed: 0.0,
        };
        self.current_state.to_array()
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        state[(0, 0)] >= 0.5
    }

    /// The state evolves with the following dynamics:
    /// $$
    /// speed_{t+1} = speed_{t} + (action - 1) * force - \cos( 3 * position_t) * gravity
    /// $$
    /// $$
    /// position_{t+1} = position_{t} + speed_{t+1}
    /// $$
    fn step<R>(
        &mut self,
        action: &Self::Action,
        _rng: &mut R,
    ) -> crate::StepResult<Self::State, Self::Action>
    where
        R: rand::Rng + ?Sized,
    {
        let starting_state = self.current_state.to_array();
        if !self.is_valid_action(action) {
            return Err(crate::EnvironmentError::InvalidAction(*action));
        }
        if !self.is_terminal(&self.current_state.to_array()) {
            let mut new_speed = self.current_state.speed + (action - 1) as f32 * self.force
                - f32::cos(3.0 * self.current_state.position) * self.gravity;
            let new_position = self.current_state.position + new_speed;

            if new_position <= self.position_bounds.0 {
                new_speed = 0.0;
            }

            self.current_state.position =
                new_position.clamp(self.position_bounds.0, self.position_bounds.1);
            self.current_state.speed = new_speed.clamp(self.speed_bounds.0, self.speed_bounds.1);
        }
        Ok(Step {
            state: starting_state,
            action: *action,
            next_state: self.current_state.to_array(),
            reward: self.get_state_reward(&self.current_state),
            terminated: self.is_terminal(&self.current_state.to_array()),
            truncated: false,
        })
    }

    fn render(&self) {
        todo!()
    }

    fn set_state(&mut self, state: &Self::State) {
        let new_state = MountainCarState::from_array(state);
        if (new_state.position > self.position_bounds.1)
            | (new_state.position < self.position_bounds.0)
            | (new_state.speed > self.speed_bounds.1)
            | (new_state.speed < self.speed_bounds.0)
        {
            panic!("Invalid state {state}")
        }
    }
}

impl DiscreteActionContinuousEnvironment<f32> for MountainCar {
    fn get_state_space(&self) -> super::ContinuousDataSpace<f32> {
        ContinuousDataSpace {
            dimensions: 2,
            range: vec![
                DataRange::<f32> {
                    low: self.position_bounds.0,
                    high: self.position_bounds.1,
                },
                DataRange::<f32> {
                    low: self.speed_bounds.0,
                    high: self.speed_bounds.1,
                },
            ],
        }
    }

    fn get_number_actions(&self) -> i32 {
        self.n_actions
    }
}

#[cfg(test)]
mod tests {

    use rand::thread_rng;

    use crate::{
        continuous::mountain_car::{MountainCarState, THROTTLE_REVERSE},
        Environment,
    };

    use super::{MountainCar, THROTTLE_FORWARD};

    #[test]
    fn test_reset() {
        let mut rng = thread_rng();
        let env = MountainCar::new(&mut rng);

        assert_eq!(env.current_state.speed, 0.0);
        assert!(env.current_state.position >= env.position_bounds.0);
        assert!(env.current_state.position <= env.position_bounds.1);
    }

    #[test]
    fn test_step_throttle_forward() {
        let mut rng = thread_rng();
        let mut env = MountainCar::new(&mut rng);

        let result = env.step(&THROTTLE_FORWARD, &mut rng);

        assert!(result.is_ok());

        let result = result.unwrap();

        assert_eq!(result.reward, -1.0);
    }

    #[test]
    fn test_reaching_goal() {
        let mut rng = thread_rng();
        let mut env = MountainCar::new(&mut rng);

        let mut action = THROTTLE_FORWARD;
        let mut success = false;
        for _ in 0..200 {
            let result = env.step(&action, &mut rng).unwrap();
            if result.terminated {
                success = true;
                break;
            }
            if MountainCarState::from_array(&result.next_state).speed > 0.0 {
                action = THROTTLE_FORWARD;
            } else {
                action = THROTTLE_REVERSE;
            }
        }
        assert!(success)
    }
}
