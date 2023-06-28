use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::tabular::generate_tabular_episode;
use rlalgs::learn::tabular::montecarlo::montecarlo;
use rlalgs::learn::tabular::n_steps::n_step_sarsa;
use rlalgs::learn::tabular::temporal_difference::{double_qlearning, qlearning, sarsa};
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlenv::tabular::cliff_walking::CliffWalking;
use rlenv::tabular::frozen::FrozenLake;
use rlenv::tabular::windy_gridworld::WindyGridworld;
use rlenv::tabular::TabularEnvironment;

#[test]
fn montecarlo_egreedy_frozen() {
    let mut rng = StdRng::seed_from_u64(222);
    let mut env = FrozenLake::new();
    let mut policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.8,
        false,
    );
    let result = montecarlo(&mut policy, &mut env, 10000, 0.999, false, false, &mut rng);
    assert!(result.is_ok());

    policy.epsilon = 0.0;
    let episode = generate_tabular_episode(&mut policy, &mut env, None, &mut rng, false).unwrap();
    assert_eq!(episode.states, vec![0, 4, 8, 9, 13, 14]);
    assert_eq!(episode.actions, vec![1, 1, 2, 1, 2, 2]);
    assert_eq!(episode.rewards, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn sarsa_windy_girdworld() {
    let mut rng = StdRng::seed_from_u64(222);
    let mut env = WindyGridworld::new();
    env.reset();

    let mut policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );
    let result = sarsa(
        &mut policy,
        &mut env,
        10000,
        40,
        0.999,
        0.5,
        false,
        false,
        &mut rng,
    );
    assert!(result.is_ok());
    policy.epsilon = 0.0;
    let episode =
        generate_tabular_episode(&mut policy, &mut env, None, &mut rand::thread_rng(), false)
            .unwrap();
    assert_eq!(
        episode.states,
        vec![30, 31, 32, 33, 24, 15, 5, 6, 7, 8, 9, 19, 29, 39, 49, 48]
    );
    assert_eq!(
        episode.actions,
        vec![2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
    );
    assert_eq!(
        episode.rewards,
        vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, 0.0
        ]
    );
}

#[test]
fn sarsa_cliff_walking() {
    let mut rng = StdRng::seed_from_u64(222);
    let mut env = CliffWalking::new();
    env.reset();

    let mut policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );
    let result = sarsa(
        &mut policy,
        &mut env,
        500,
        5000,
        1.0,
        0.5,
        false,
        false,
        &mut rng,
    );
    assert!(result.is_ok());
    policy.epsilon = 0.0;
    let episode =
        generate_tabular_episode(&mut policy, &mut env, None, &mut rand::thread_rng(), false)
            .unwrap();
    assert_eq!(
        episode.states,
        vec![30, 20, 10, 0, 1, 2, 3, 4, 5, 6, 7, 17, 18, 28, 29]
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
    let mut env = CliffWalking::new();
    env.reset();
    env.render();

    let mut policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );
    let result = qlearning(&mut policy, &mut env, 500, 5000, 1.0, 0.5, false, &mut rng);
    assert!(result.is_ok());
    policy.epsilon = 0.0;
    let episode =
        generate_tabular_episode(&mut policy, &mut env, None, &mut rand::thread_rng(), false)
            .unwrap();
    assert_eq!(
        episode.states,
        vec![30, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
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
    let mut env = CliffWalking::new();
    env.reset();

    let mut policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );
    let result = sarsa(
        &mut policy,
        &mut env,
        500,
        50000,
        1.0,
        0.5,
        true,
        false,
        &mut rng,
    );

    policy.epsilon = 0.0;
    let episode = generate_tabular_episode(
        &mut policy,
        &mut env,
        Some(50),
        &mut rand::thread_rng(),
        true,
    )
    .unwrap();

    assert!(result.is_ok());
    assert_eq!(
        episode.states,
        vec![30, 20, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29]
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
    let mut env = CliffWalking::new();
    env.reset();
    env.render();

    let mut policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );
    let result = double_qlearning(&mut policy, &mut env, 500, 5000, 1.0, 0.5, false, &mut rng);
    assert!(result.is_ok());
    policy.epsilon = 0.0;
    let episode =
        generate_tabular_episode(&mut policy, &mut env, None, &mut rand::thread_rng(), false)
            .unwrap();
    assert_eq!(
        episode.states,
        vec![30, 20, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 29]
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
    let mut env = CliffWalking::new();
    env.reset();

    let mut policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );
    let result = n_step_sarsa(
        &mut policy,
        &mut env,
        500,
        20,
        1.0,
        0.5,
        false,
        false,
        &mut rng,
    );
    assert!(result.is_ok());
    policy.epsilon = 0.0;
    let episode =
        generate_tabular_episode(&mut policy, &mut env, None, &mut rand::thread_rng(), false)
            .unwrap();
    assert_eq!(
        episode.states,
        vec![30, 20, 10, 11, 21, 22, 12, 13, 3, 4, 5, 6, 7, 8, 9, 19, 29]
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
    let mut env = CliffWalking::new();
    env.reset();

    let mut policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );
    let result = n_step_sarsa(
        &mut policy,
        &mut env,
        500,
        20,
        1.0,
        0.5,
        true,
        false,
        &mut rng,
    );
    assert!(result.is_ok());
    policy.epsilon = 0.0;
    let episode =
        generate_tabular_episode(&mut policy, &mut env, None, &mut rand::thread_rng(), false)
            .unwrap();
    assert_eq!(
        episode.states,
        vec![30, 20, 21, 11, 1, 2, 3, 13, 14, 4, 5, 6, 7, 8, 9, 19, 29]
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
