use std::cell::RefCell;

use rlalgs::generate_episode;
use rlalgs::learn::VerbosityConfig;
use rlalgs::model::{Model, ModelStep};
use rlalgs::policy::tabular::egreedy::EGreedyTabularPolicy;
use rlalgs::policy::tabular::mcts;
use rlenv::tabular::terror_maze::TerrorMaze;
use rlenv::tabular::TabularEnvironment;
use rlenv::Environment;

struct TerrorMazeModelWrapper {
    env: RefCell<TerrorMaze>,
}

impl Model for TerrorMazeModelWrapper {
    type State = i32;

    type Action = i32;

    fn predict_step(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> rlalgs::model::ModelStep<Self::State> {
        self.env.borrow_mut().set_state(state);
        let env_step = self
            .env
            .borrow_mut()
            .step(action, &mut rand::thread_rng())
            .unwrap();
        ModelStep {
            state: env_step.next_state,
            reward: env_step.reward,
        }
    }

    fn update_step(
        &mut self,
        state: &Self::State,
        action: &Self::Action,
        next_state: &Self::State,
        reward: f32,
    ) {
        todo!()
    }

    fn sample_sa<R>(
        &self,
        rng: &mut R,
    ) -> Option<rlalgs::model::SampleSA<Self::State, Self::Action>>
    where
        R: rand::Rng + ?Sized,
    {
        todo!()
    }

    fn get_preceding_sa(
        &self,
        state: &Self::State,
    ) -> Option<&Vec<rlalgs::StateAction<Self::State, Self::Action>>> {
        todo!()
    }
}

#[test]
fn mcts_test() {
    // Create environment
    let env = TerrorMaze::new();

    let verbosity = VerbosityConfig {
        render_env: false,
        episode_progress: false,
    };

    let rollout_policy = EGreedyTabularPolicy::new(
        env.get_number_states() as usize,
        env.get_number_actions() as usize,
        0.1,
        true,
    );

    // As model we use the actual true environment. This is a wrapper that
    // contains the true environment
    let model = TerrorMazeModelWrapper {
        env: RefCell::new(TerrorMaze::new()),
    };

    let env_essay = TerrorMaze::new();

    // Create policy
    let mut policy = mcts::MCTSPolicy::new(
        mcts::UCBSelector::new(1.0),
        rollout_policy,
        model,
        10000,
        300,
        mcts::RootActionCriterion::HighestScore,
        false,
        verbosity,
        env_essay,
        0.999,
    );

    // Make an episode with greedy policy
    let mut env = TerrorMaze::new();
    let episode = generate_episode(
        &mut policy,
        &mut env,
        None,
        &mut rand::thread_rng(),
        true,
        None,
    )
    .unwrap();
    assert_eq!(env.is_terminal(episode.states.last().unwrap()), true);
    assert_eq!(*episode.states.last().unwrap(), 20);
}
