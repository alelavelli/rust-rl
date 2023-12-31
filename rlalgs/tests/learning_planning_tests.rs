use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::generate_episode;
use rlalgs::learn::planning::tabular::{dyna_q, prioritized_sweeping};
use rlalgs::learn::VerbosityConfig;
use rlalgs::model::tabular::deterministic::DeterministicModel;
use rlalgs::policy::egreedy::EGreedyPolicy;
use rlenv::tabular::simple_maze::SimpleMaze;
use rlenv::tabular::TabularEnvironment;
use rlenv::Environment;

#[test]
fn dyna_q_test() {
    let mut rng = StdRng::seed_from_u64(222);

    // Create environment
    let env = SimpleMaze::new();

    // Create policy
    let policy = EGreedyPolicy::new_discrete(
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
    let params = dyna_q::Params {
        n_iterations: 1000,
        real_world_steps: 1,
        simulation_steps: 50,
        gamma: 0.95,
        step_size: 0.1,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        learning_progress: false,
        episode_progress: false,
    };

    // Learn policy
    let result = dyna_q::learn(policy, env, model, params, &mut rng, &verbosity);

    // Make an episode with greedy policy
    let mut env = SimpleMaze::new();
    let (mut policy, mut _model) = result.unwrap();
    policy.set_epsilon(0.0);
    let episode = generate_episode(
        &mut policy,
        &mut env,
        None,
        &mut rand::thread_rng(),
        true,
        None,
    )
    .unwrap();
    assert_eq!(env.is_terminal(episode.states.last().unwrap()), true)
}

#[test]
fn prioritized_sweeping_test() {
    let mut rng = StdRng::seed_from_u64(222);

    // Create environment
    let env = SimpleMaze::new();

    // Create policy
    let policy = EGreedyPolicy::new_discrete(
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
        learning_progress: false,
        episode_progress: false,
    };

    // Learn policy
    let result = prioritized_sweeping::learn(policy, env, model, params, &mut rng, &verbosity);

    // Make an episode with greedy policy
    let mut env = SimpleMaze::new();
    let (mut policy, mut _model) = result.unwrap();
    policy.set_epsilon(0.0);
    let episode = generate_episode(
        &mut policy,
        &mut env,
        None,
        &mut rand::thread_rng(),
        true,
        None,
    )
    .unwrap();
    assert_eq!(env.is_terminal(episode.states.last().unwrap()), true)
}
