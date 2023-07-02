use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::tabular::generate_tabular_episode;
use rlalgs::learn::tabular::montecarlo;
use rlalgs::learn::VerbosityConfig;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlalgs::policy::tabular::TabularPolicy;
use rlenv::tabular::frozen::FrozenLake;
use rlenv::tabular::TabularEnvironment;

fn main() {
    let mut rng = StdRng::seed_from_u64(222);
    let env = FrozenLake::new();

    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        false,
    );

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let params = montecarlo::Params {
        episodes: 100,
        gamma: 1.0,
        first_visit_mode: false,
    };

    let result = montecarlo::learn(policy, env, params, &mut rng, &verbosity);
    let mut policy = result.unwrap();
    let mut env = FrozenLake::new();
    println!("{:^20?}", policy.get_q());

    policy.set_epsilon(0.0);
    let episode = generate_tabular_episode(
        &mut policy,
        &mut env,
        Some(50),
        &mut rand::thread_rng(),
        true,
        None,
    );
    println!("{:?}", episode);
}
