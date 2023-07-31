use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::generate_episode;
use rlalgs::learn::VerbosityConfig;
use rlalgs::model::tabular::deterministic::DeterministicModel;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlalgs::policy::tabular::mcts;
use rlenv::tabular::simple_maze::SimpleMaze;
use rlenv::tabular::TabularEnvironment;
use rlenv::Environment;

#[test]
fn mcts_test() {
    let mut rng = StdRng::seed_from_u64(222);

    // Create environment
    let env = SimpleMaze::new();

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let rollout_policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let model = DeterministicModel::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
    );

    // Create policy
    let policy = mcts::MCTSPolicy::new(
        mcts::UCBSelector::new(1.0),
        rollout_policy,
        model,
        10000,
        300,
        mcts::RootActionCriterion::HighestScore,
        false,
        verbosity,
        env_essay,
        0.999
    );

    // Make an episode with greedy policy
    let mut env = SimpleMaze::new();
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
