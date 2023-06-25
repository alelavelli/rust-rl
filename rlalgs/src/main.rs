use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::tabular::generate_tabular_episode;
use rlalgs::learn::tabular::temporal_difference::qlearning;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlenv::tabular::cliff_walking::CliffWalking;
use rlenv::tabular::TabularEnvironment;

fn main() {
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
    let result = qlearning(&mut policy, &mut env, 10000, 40, 1.0, 0.5, false, &mut rng);
    println!("{:^20?}", policy.q);

    policy.epsilon = 0.0;
    let episode = generate_tabular_episode(&mut policy, &mut env, &mut rand::thread_rng(), true);
    println!("{:?}", episode);
    assert!(result.is_ok());
}
