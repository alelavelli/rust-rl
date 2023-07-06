use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use rand::Rng;
use rlenv::tabular::TabularEnvironment;

use crate::{
    learn::{LearningError, VerbosityConfig},
    policy::tabular::TabularPolicy,
};

/// Parameters for dyna-q learning algorithm
///
/// - `n_iterations`: number of iteration of the algorithm. 
/// An iteration consists of making a real environment step 
/// and n simulated steps
/// - `simulation_steps`: number of simulation steps
/// - `gamma`: discount factor
/// - `step_size`: step size of the update rule
pub struct Params {
    pub n_iterations: i32,
    pub simulation_steps: i32,
    pub gamma: f32,
    pub step_size: f32,
}

/// Dyna-Q model based learning algorithm
/// $$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right] $$
///
/// In each iteration of Dyna-Q algorithm, the agent takes one real environment step
/// and then n simulated steps. The policy is updated at each step with Q learning algorithm
/// 
/// ## Parameters
///
/// - `policy`: TabularPolicy to learn
/// - `environment`: TabularEnvironment
/// - `model`: TabularModel
/// - `params`: algorithm parameters
/// - `rng`: random generator
/// - `verbosity`: verbosity configuration
///
#[allow(clippy::too_many_arguments)]
pub fn learn<P, E, R, M>(
    mut policy: P,
    mut environment: E,
    mut model: M,
    direct_learning: Box<dyn Fn(P, &mut E) -> P>,
    planning_learning: Box<dyn Fn(P, M) -> (P, M)>,
    params: Params,
    rng: &mut R,
    versbosity: &VerbosityConfig,
) -> Result<(P, M), LearningError>
where
    P: TabularPolicy,
    E: TabularEnvironment,
    R: Rng + ?Sized,
    M: TabularModel
{
    
}
