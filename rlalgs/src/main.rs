use std::ops::Div;

use ndarray::{array, s, ArrayView, Ix1};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rlalgs::learn::tabular::generate_tabular_episode;
use rlalgs::learn::tabular::temporal_difference::sarsa;
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlalgs::policy::tabular::TabularPolicy;
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

    // update q entry using weighted q value
    let a_probs = array![0.2, 0.4, 0.1, 0.0];
    println!("{}", a_probs);
    let q_row: ArrayView<_, Ix1> = policy.get_q().slice(s![0, ..]);
    let q_expected = q_row.dot(&a_probs).div(4.0);
    println!("{}", q_row);
    println!("{}", q_expected);
}
