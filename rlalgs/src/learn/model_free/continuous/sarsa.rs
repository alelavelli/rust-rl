use ndarray::{Array1, Array2};
use rand::Rng;
use rlenv::{continuous::DiscreteActionContinuousEnvironment, Environment};

use crate::{
    learn::{LearningError, VerbosityConfig},
    policy::Policy,
    value_function::StateActionValueFunction,
};

/// Parameters for sarsa learning algorithm
///
/// `episodes`: number of episodes to generate during learning
/// `episode_max_len`: maximum length of a single episode before stopping it
/// `gamma`: discount factor
/// `step_size`: step size of the updating rule
pub struct Params {
    pub episodes: i32,
    pub episode_max_len: i32,
    pub gamma: f32,
    pub step_size: f32,
    pub expected: bool,
}

/// Sarsa on-policy TD with contunous state
///
/// Update rule:
/// $$ \boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha \left[  R + \gamma \hat{q}(S_{t+1}, A_{t+1}, \boldsymbol{w}) - \hat{q}(S_{t}, A_{t}, \boldsymbol{w}) \right] \nabla \hat{q}(S_{t}, A_{t}, \boldsymbol{w}) $$
pub fn learn<P, E, R>(
    mut policy: P,
    mut environment: E,
    params: Params,
    rng: &mut R,
    verbosity: &VerbosityConfig,
) -> Result<P, Box<LearningError<Array2<f32>, i32>>>
where
    P: Policy<State = Array1<f32>, Action = i32> + StateActionValueFunction,
    E: Environment<State = Array2<f32>, Action = i32> + DiscreteActionContinuousEnvironment,
    R: Rng + ?Sized,
{
    Ok(policy)
}
