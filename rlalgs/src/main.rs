use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::tabular::generate_tabular_episode;
use rlalgs::learn::tabular::sarsa;
use rlalgs::learn::VerbosityConfig;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlenv::tabular::windy_gridworld::WindyGridworld;
use rlenv::tabular::TabularEnvironment;

fn main() {
    let mut rng = StdRng::seed_from_u64(222);
    
    // Create environment
    let env = WindyGridworld::new();

    // Create policy
    let policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    // Define parameters
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

    // Learn policy
    let result = sarsa::learn(policy, env, params, &mut rng, &verbosity);
    
    // Make an episode with greedy policy
    let mut env = WindyGridworld::new();
    let mut policy = result.unwrap();
    policy.set_epsilon(0.0);
    let episode = generate_tabular_episode(
        &mut policy,
        &mut env,
        None,
        &mut rand::thread_rng(),
        true,
        None,
    );
}