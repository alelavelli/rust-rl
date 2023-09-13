# rust-rl
Implementation of Reinforcement Algorithms with Rust.

The approach I chosen is to follow Sutton & Barto Reinforcement Learning Book implementing each algorithm they explain.
I know that all these algorithms are not used in industry world but they can be a useful starting point to learn both Reinforcement Learning and Rust.

The repository is structured as a workspace with two packages:
- `rlalgs`: library crate with learning algorithms and policies
- `rlenv`: library crate with enviroment traits

# `rlalgs` package

The package is composed by two main modules: `learn` and `policy`. `learn` module has the implementation learning algorithms that given a policy and an environment returns a policy that learnt to behave on the environment. `policy` module has the implementation of different kind of policies.

Both modules, follow the Reinforcement Learning tassonomy, hence, they split in tabular, continuous and so on.

# `rlenv` package

The package provides trait an environment must have to be used by the `rlalgs` package. It contains toys environments used for testing the algorithms and the policies. It has also the feature `gymnasium` that through Python integrations allows to use all the gymnasium environments.

# Implementations
Here, all the implemented parts.

## Policies

- epsilon greedy policy

## Learning Algorithms

### Tabular

- montecarlo every visit and first visit
- sarsa
- expected sarsa
- qlearning
- double qlearning
- n step sarsa

# Example

```rs
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
        learning_progress: false,
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
```
