use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::tabular::generate_tabular_episode;
use rlalgs::learn::tabular::montecarlo::montecarlo;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlenv::tabular::frozen::FrozenLake;
use rlenv::tabular::TabularEnvironment;

fn main() {
    let mut rng = StdRng::seed_from_u64(222);
    let mut env = FrozenLake::new();
    let mut policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.8,
    );
    let result = montecarlo(&mut policy, &mut env, 10000, 0.999, false, false, &mut rng);
    println!("{:^20?}", policy.q);

    policy.epsilon = 0.0;
    let episode = generate_tabular_episode(&mut policy, &mut env, &mut rand::thread_rng(), true);
    println!("{:?}", episode);
    assert!(result.is_ok());
}
