use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::model_free::tabular::generate_tabular_episode;
use rlalgs::learn::model_free::tabular::n_step_q_sigma;
use rlalgs::learn::VerbosityConfig;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlenv::tabular::simple_maze::SimpleMaze;
use rlenv::tabular::TabularEnvironment;

fn main() {
    let mut rng = StdRng::seed_from_u64(222);

    // Create environment
    let env = SimpleMaze::new();

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

    // Learn policy
    let result = n_step_q_sigma::learn(policy, behaviour, env, params, &mut rng, &verbosity);

    // Make an episode with greedy policy
    let mut env = SimpleMaze::new();
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
    println!("{:?}", episode);
}
