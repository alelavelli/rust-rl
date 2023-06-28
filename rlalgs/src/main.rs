use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::tabular::generate_tabular_episode;
use rlalgs::learn::tabular::n_steps::n_step_sarsa;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlenv::tabular::cliff_walking::CliffWalking;
use rlenv::tabular::TabularEnvironment;

fn main() {
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
    println!("{:^20?}", policy.q);

    policy.epsilon = 0.0;
    let episode = generate_tabular_episode(
        &mut policy,
        &mut env,
        Some(50),
        &mut rand::thread_rng(),
        true,
    );
    println!("{:?}", episode);
    assert!(result.is_ok());
}
