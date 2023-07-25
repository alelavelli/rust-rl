use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::generate_episode;
use rlalgs::learn::model_free::tabular::double_qlearning;
use rlalgs::learn::model_free::tabular::montecarlo;
use rlalgs::learn::model_free::tabular::n_step_q_sigma;
use rlalgs::learn::model_free::tabular::n_step_sarsa;
use rlalgs::learn::model_free::tabular::n_step_tree_backup;
use rlalgs::learn::model_free::tabular::qlearning;
use rlalgs::learn::model_free::tabular::sarsa;
use rlalgs::learn::VerbosityConfig;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlenv::tabular::cliff_walking::CliffWalking;
use rlenv::tabular::frozen::FrozenLake;
use rlenv::tabular::windy_gridworld::WindyGridworld;
use rlenv::tabular::TabularEnvironment;

#[test]
fn montecarlo_egreedy_frozen() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = FrozenLake::new();
    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.8,
        false,
    );
    let params = montecarlo::Params {
        episodes: 2000,
        gamma: 1.0,
        first_visit_mode: false,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = montecarlo::learn(policy, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();

    policy.set_epsilon(0.0);
    let mut env = FrozenLake::new();
    let episode =
        generate_episode(&mut policy, &mut env, Some(10), &mut rng, true, None).unwrap();
    assert_eq!(episode.states, vec![0, 4, 8, 9, 13, 14, 15]);
    assert_eq!(episode.actions, vec![1, 1, 2, 1, 2, 2]);
    assert_eq!(episode.rewards, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn sarsa_windy_girdworld() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = WindyGridworld::new();

    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let params = sarsa::Params {
        episodes: 500,
        episode_max_len: 100,
        gamma: 0.999,
        step_size: 0.5,
        expected: false,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = sarsa::learn(policy, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();
    let mut env = WindyGridworld::new();
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
    assert_eq!(
        episode.states,
        vec![30, 31, 32, 33, 24, 15, 6, 7, 8, 9, 19, 29, 39, 49, 48, 37]
    );
    assert_eq!(
        episode.actions,
        vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
    );
    assert_eq!(
        episode.rewards,
        vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0
        ]
    );
}

#[test]
fn sarsa_cliff_walking() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = CliffWalking::new();

    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let params = sarsa::Params {
        episodes: 500,
        episode_max_len: 5000,
        gamma: 1.0,
        step_size: 0.5,
        expected: false,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = sarsa::learn(policy, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();
    let mut env = CliffWalking::new();
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
    assert_eq!(
        episode.states,
        vec![30, 20, 10, 0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 28, 29, 39]
    );
    assert_eq!(
        episode.actions,
        vec![3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1]
    );
    assert_eq!(
        episode.rewards,
        vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0
        ]
    );
}

#[test]
fn qlearning_cliff_walking() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = CliffWalking::new();

    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let params = qlearning::Params {
        episodes: 500,
        episode_max_len: 5000,
        gamma: 1.0,
        step_size: 0.5,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = qlearning::learn(policy, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();
    let mut env = CliffWalking::new();
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
    assert_eq!(
        episode.states,
        vec![30, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 39]
    );
    assert_eq!(episode.actions, vec![3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]);
    assert_eq!(
        episode.rewards,
        vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
    );
}

#[test]
fn expected_sarsa_cliff_walking() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = CliffWalking::new();

    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let params = sarsa::Params {
        episodes: 500,
        episode_max_len: 5000,
        gamma: 1.0,
        step_size: 0.5,
        expected: true,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = sarsa::learn(policy, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();
    let mut env = CliffWalking::new();
    policy.set_epsilon(0.0);
    let episode = generate_episode(
        &mut policy,
        &mut env,
        Some(50),
        &mut rand::thread_rng(),
        true,
        None,
    )
    .unwrap();

    assert_eq!(
        episode.states,
        vec![30, 20, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29, 39]
    );
    assert_eq!(episode.actions, vec![3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]);
    assert_eq!(
        episode.rewards,
        vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
    );
}

#[test]
fn double_qlearning_cliff_walking() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = CliffWalking::new();

    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let params = double_qlearning::Params {
        episodes: 500,
        episode_max_len: 5000,
        gamma: 1.0,
        step_size: 0.5,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = double_qlearning::learn(policy, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();
    let mut env = CliffWalking::new();
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
    assert_eq!(
        episode.states,
        vec![30, 20, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29, 39]
    );
    assert_eq!(
        episode.actions,
        vec![3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
    );
    assert_eq!(
        episode.rewards,
        vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0
        ]
    );
}

#[test]
fn n_step_sarsa_cliff_walking() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = CliffWalking::new();

    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let params = n_step_sarsa::Params {
        episodes: 500,
        n: 20,
        gamma: 1.0,
        step_size: 0.5,
        expected: false,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = n_step_sarsa::learn(policy, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();
    let mut env = CliffWalking::new();
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
    assert_eq!(
        episode.states,
        vec![30, 20, 10, 11, 21, 22, 12, 13, 3, 4, 5, 6, 7, 8, 9, 19, 29, 39]
    );
    assert_eq!(
        episode.actions,
        vec![3, 3, 2, 1, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1]
    );
    assert_eq!(
        episode.rewards,
        vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, 0.0
        ]
    );
}

#[test]
fn n_step_expected_sarsa_cliff_walking() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = CliffWalking::new();

    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let params = n_step_sarsa::Params {
        episodes: 500,
        n: 20,
        gamma: 1.0,
        step_size: 0.5,
        expected: true,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = n_step_sarsa::learn(policy, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();
    let mut env = CliffWalking::new();
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
    assert_eq!(
        episode.states,
        vec![30, 20, 21, 11, 1, 2, 3, 13, 14, 4, 5, 6, 7, 8, 9, 19, 29, 39]
    );
    assert_eq!(
        episode.actions,
        vec![3, 2, 3, 3, 2, 2, 1, 2, 3, 2, 2, 2, 2, 2, 1, 1, 1]
    );
    assert_eq!(
        episode.rewards,
        vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, 0.0
        ]
    );
}

#[test]
fn n_step_tree_backup_windy_girdworld() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = WindyGridworld::new();

    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    let params = n_step_tree_backup::Params {
        episodes: 200,
        gamma: 0.999,
        step_size: 0.5,
        expected: false,
        n: 10,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = n_step_tree_backup::learn(policy, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();
    let mut env = WindyGridworld::new();
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
    assert_eq!(
        episode.states,
        vec![30, 31, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9, 19, 29, 39, 49, 48, 37]
    );
    assert_eq!(
        episode.actions,
        vec![2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
    );
    assert_eq!(
        episode.rewards,
        vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, 0.0
        ]
    );
}

#[test]
fn n_step_q_sigma_windy_girdworld() {
    let mut rng = StdRng::seed_from_u64(222);

    // Create environment
    let env = WindyGridworld::new();

    // Create policy
    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.5,
        true,
    );

    let behaviour = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.8,
        false,
    );

    fn sigma_fn(input: n_step_q_sigma::SigmaInput) -> f32 {
        (input.step % 2) as f32
    }

    // Define parameters
    let sigma_fn_box = Box::new(sigma_fn);
    let params = n_step_q_sigma::Params {
        episodes: 500,
        gamma: 0.9,
        step_size: 0.5,
        sigma_fn: sigma_fn_box,
        update_behaviour: true,
        n: 10,
    };

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let result = n_step_q_sigma::learn(policy, behaviour, env, params, &mut rng, &verbosity);
    assert!(result.is_ok());
    let mut policy = result.unwrap();
    let mut env = WindyGridworld::new();
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
    assert_eq!(
        episode.states,
        vec![30, 31, 32, 33, 24, 15, 6, 7, 8, 9, 19, 29, 39, 49, 48, 37]
    );
    assert_eq!(
        episode.actions,
        vec![2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
    );
    assert_eq!(
        episode.rewards,
        vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0
        ]
    );
}
