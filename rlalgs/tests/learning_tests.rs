use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::tabular::generate_tabular_episode;
use rlalgs::learn::tabular::montecarlo::montecarlo;
use rlalgs::learn::tabular::temporal_difference::sarsa;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
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
    println!("{:^20?}", policy.q);
    assert!(result.is_ok());

    policy.epsilon = 0.0;
    let episode = generate_tabular_episode(&mut policy, &mut env, &mut rng, true).unwrap();
    println!("{:?}", episode);
    assert_eq!(episode.states, vec![0, 4, 8, 9, 13, 14]);
    assert_eq!(episode.actions, vec![1, 1, 2, 1, 2, 2]);
    assert_eq!(episode.rewards, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn sarsa_windy_girdworld() {
    let mut rng = StdRng::seed_from_u64(222);
    let mut env = WindyGridworld::new();
    env.reset();
    env.render();

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
        &mut rng,
    );
    println!("{:^20?}", policy.q);

    policy.epsilon = 0.0;
    let episode =
        generate_tabular_episode(&mut policy, &mut env, &mut rand::thread_rng(), true).unwrap();
    println!("{:?}", episode);
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
