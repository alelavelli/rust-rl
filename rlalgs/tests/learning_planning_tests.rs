use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::model_free::tabular::generate_tabular_episode;
use rlalgs::learn::planning::tabular::dyna_q;
use rlalgs::learn::VerbosityConfig;
use rlalgs::model::tabular::deterministic::DeterministicModel;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlenv::tabular::simple_maze::SimpleMaze;
use rlenv::tabular::TabularEnvironment;

#[test]
fn dyna_q_test() {
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
    let params = dyna_q::Params {
        n_iterations: 1000,
        real_world_steps: 1,
        simulation_steps: 50,
        gamma: 0.95,
        step_size: 0.1,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    // Learn policy
    let result = dyna_q::learn(policy, env, model, params, &mut rng, &verbosity);

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
    )
    .unwrap();
    assert_eq!(
        episode.states,
        vec![18, 27, 36, 37, 38, 39, 30, 31, 32, 33, 34, 35, 26, 17]
    );
    assert_eq!(
        episode.actions,
        vec![1, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 3, 3]
    );
    assert_eq!(
        episode.rewards,
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
}