/* use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::model_free::tabular::generate_tabular_episode;
use rlalgs::learn::planning::tabular::prioritized_sweeping;
use rlalgs::learn::VerbosityConfig;
use rlalgs::model::tabular::deterministic::DeterministicModel;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlenv::tabular::simple_maze::SimpleMaze;
use rlenv::tabular::TabularEnvironment;

fn main() {
    let mut rng = StdRng::seed_from_u64(222);

    // Create environment
    let env = SimpleMaze::new();

    // Create policy
    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let model = DeterministicModel::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
    );

    // define parameters
    let params = prioritized_sweeping::Params {
        n_iterations: 8000,
        simulation_steps: 5,
        tolerance: 0.05,
        gamma: 0.95,
        step_size: 0.5,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    // Learn policy
    let result = prioritized_sweeping::learn(policy, env, model, params, &mut rng, &verbosity);

    // Make an episode with greedy policy
    let mut env = SimpleMaze::new();
    let (mut policy, mut _model) = result.unwrap();
    policy.set_epsilon(0.0);
    let episode = generate_tabular_episode(
        &mut policy,
        &mut env,
        None,
        &mut rand::thread_rng(),
        true,
        None,
    );
    println!("{:?}", episode);
}
 */

 fn main() {
    
 }