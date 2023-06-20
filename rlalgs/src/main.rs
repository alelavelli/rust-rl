use colored::Colorize;
use rand::seq::SliceRandom;
use rlenv::tabular::TabularEnvironment;
fn main() {
    rlenv::hello_free();

    let mut env = rlenv::tabular::frozen::FrozenLake::new();
    let mut rng = rand::thread_rng();
    let actions = vec![0, 1, 2, 3];
    env.render();
    for i in 0..10 {
        let a = *actions.choose(&mut rng).unwrap();
        println!("Action: {}", a);
        _ = env.step(a, &mut rng);
        env.render();
    }
}
