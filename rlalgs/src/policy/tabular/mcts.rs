use std::{
    error::Error,
    fmt::Debug,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use ndarray::Array1;
use ndarray_stats::QuantileExt;
use rand::Rng;
use rlenv::{DiscreteActionEnvironmentEssay, EnvironmentEssay};

use crate::{
    learn::VerbosityConfig,
    model::Model,
    policy::Policy,
    utils::arena_tree::{self, TreeArena},
};

/// Attributes of the MCTS node that is used as value inside the Tree data structure
///
/// - `state`: state represented by the node
/// - `actions`: available actions from the node
/// - `terminal`: if it represents a terminal state
/// - `parent_source_action`: parent's action that led to this node
/// - `visits`: number of visits to this node during mcts iteraction
/// - `value`: value of this node
/// - `action_visits`: number of visits for each available action
/// - `action_values`: value associated to each avaiable action
pub struct NodeAttributes<S, A> {
    state: S,
    actions: Vec<A>,
    terminal: bool,
    parent_source_action: Option<A>,
    visits: u32,
    value: f32,
    action_visits: Array1<u32>,
    action_values: Array1<f32>,
}

impl<S, A> NodeAttributes<S, A> {
    fn new(
        state: S,
        actions: Vec<A>,
        terminal: bool,
        parent_source_action: Option<A>,
    ) -> NodeAttributes<S, A> {
        let n_actions = actions.len();
        NodeAttributes {
            state,
            actions,
            terminal,
            parent_source_action,
            visits: 0,
            value: 0.0,
            action_visits: Array1::zeros(n_actions),
            action_values: Array1::zeros(n_actions),
        }
    }
}

pub enum RootActionCriterion {
    MostPlayed,
    HisghestScore,
}

/// Monte Carlo Tree Search for Tabular Data
///
/// Implementation of MCTS for tabular data focusing on high parallelization
/// of the iterations.
///
/// The method is not `learn` but choose_action because it does not learn a policy
/// but uses all the information it has to choose the next action in the state.
///
/// With MCTS there are three possibilities for parallelization:
///
/// 1. **Leaf parallelization** is the simplest form of parallelization
/// and applies it in the phase 3 of MCTS algorithm. The main thread executes
/// phase 1 (selection) and phase 2 (expansion), then when a leaf is reached
/// multiple parallel montecarlo simulations are run. Finally, the phase 4
/// (back propagation) is done sequentially. This kind of parallelization is
/// useful when the time required to run a single simulation is long.
///
/// 2. **Root parallelization** consists of building multiple MCTS trees in
/// parallel, with one thread per tree. When, the time isspent, all the root children
/// of the separate MCTS trees are merged. The best move is selected based on summing
/// up the scores.
///
/// 3. **Tree Parallelization** uses one shared tree from which several simultaneous games
/// are played. Each thread can modify the information contained in the tree. Therefore,
/// mutexes are used to lock from time to time certain parts of the tree to prevent data
/// corruption. Tree parallelization has two methods to improve its performace:
///     - **mutex location**: based on the location of the mutexes in the tree, there are
///        two location methods named *global mutex* and *local mutexes*.
///     - **virtual loss**: a viartual loss is assigned when a node is visided by a thread.
///         Hence, the value of this node will be decreased. This is used to avoid multiple
///         threads visit the same portion of the tree.
///
/// # Parameters
///
/// - `tree_policy`: policy that chooses actions inside the tree
/// - `rollout_policy`: policy used during the rollout phase
/// - `model`: transition model used during rollout
/// - `iterations`: number of iterations before returning the chosen action
/// - `max_depth`: the maximum horizon to simulate forward steps
///     (i.e., the maximum depth of the tree)
/// - `root_action_criterion`: choose the most played action at the root
///     or the one with highest score. It is an Enum
/// - `reuse_tree`: whether to start MCTS from the old sub-tree whenever the new state is close to its old prediction
/// - `verbosity`: verbosity configuration struct
/// - `tree`: internal tree that is built during the iterations
/// - `env_essay`: component that contains information about the environment like is a state is terminal or what are the
/// available action in that given state
/// - `gamma`: discount factor
pub struct MCTSPolicy<T, P, M, S, A, E>
where
    T: NodeSelector<State = S, Action = A>,
    P: Policy<State = S, Action = A>,
    M: Model<State = S, Action = A>,
    E: EnvironmentEssay<State = S, Action = A>
        + DiscreteActionEnvironmentEssay<State = S, Action = A>,
{
    tree_policy: T,
    rollout_policy: P,
    model: M,
    iterations: i32,
    max_depth: i32,
    root_action_criterion: RootActionCriterion,
    reuse_tree: bool,
    verbosity: VerbosityConfig,
    tree: TreeArena<NodeAttributes<S, A>>,
    env_essay: E,
    gamma: f32
}

impl<T, P, M, S, A, E> MCTSPolicy<T, P, M, S, A, E>
where
    T: NodeSelector<State = S, Action = A>,
    P: Policy<State = S, Action = A>,
    M: Model<State = S, Action = A>,
    E: EnvironmentEssay<State = S, Action = A>
        + DiscreteActionEnvironmentEssay<State = S, Action = A>,
{
    /// Selection phase
    ///
    /// Starting at the root node, a tree policy based on the action values
    /// attached to the edges of the tree traverses the tree to select a leaf node.
    ///
    /// It contains also the *Expansion* phase in which the tree is expanded from the
    /// selected leaf node by adding one or more child nodes reached from the selected
    /// node via unexplored actions.
    ///
    /// # Parameters
    ///
    /// - `node`: where start the tree traversal. Note that this could be any node and
    /// not the root of the tree
    fn selection(
        &self,
        node: Arc<RwLock<arena_tree::Node<NodeAttributes<S, A>>>>,
    ) -> Arc<RwLock<arena_tree::Node<NodeAttributes<S, A>>>> {
        let mut current = Arc::clone(&node);
        // We loop until we reach maximum depth of the tree or if the node is a terminal one
        while {
            let read_guard = current.read().unwrap();
            (read_guard.depth < self.max_depth) & !read_guard.attributes.terminal
        } {
            // we use the tree policy to select the action and the model to compute the next state
            // at first, we check for expansions by choosing non played actions
            let expansion_action = self.tree_policy.expansion_action(current).unwrap();

            let (action, actual_expansion) = {
                if let Some(action) = expansion_action {
                    (action, true)
                } else {
                    (self.tree_policy.select_action(current).unwrap(), false)
                }
            };

            let model_step = self
                .model
                .predict_step(current.read().unwrap().attributes.state, action);

            // we add the new tree to the arena as a child of node
            let new_node_id = self
                .tree
                .add_node(
                    Some(current.read().unwrap().id),
                    NodeAttributes::new(
                        model_step.state,
                        self.env_essay.available_actions(&model_step.state),
                        self.env_essay.is_terminal(&model_step.state),
                        Some(action),
                    ),
                )
                .unwrap();

            current = self.tree.get_node(new_node_id).unwrap();
            // If the chosen action has no visits because of expansions we terminate the loop
            // and we return that node
            if actual_expansion {
                break;
            }
        }

        current
    }

    /// Simulation phase
    ///
    /// From the selected node, or from one of its newly-added child nodes, simulation of
    /// a complete episode is run with actions selected by the rollout policy. The result
    /// is a Monte Carlo trial with actions selected first by the tree policy and beyond
    /// the tree by the rollout policy
    fn rollout<R>(&self, node: Arc<RwLock<arena_tree::Node<NodeAttributes<S, A>>>>, rng: &mut R) -> f32 
    where
        R: Rng + ?Sized
        {
        // if the node is a terminal one there is nothing to do, so we return 0.0 as return
        if node.read().unwrap().attributes.terminal {
            0.0
        } else {
            let (
                mut current_state,
                mut is_terminal,
                mut current_depth
            ) = {
                let read_guard = node.read().unwrap();
                (
                    read_guard.attributes.state,
                    read_guard.attributes.terminal,
                    read_guard.depth
                )
            };
            
            let mut ret = 0.0;
            while (current_depth < self.max_depth) & !is_terminal
            {
                let action = self.rollout_policy.step(current_state, rng).unwrap();
                let model_step = self.model.predict_step(current_state, action);
                current_depth += 1;
                current_state = model_step.state;
                is_terminal = self.env_essay.is_terminal(&current_state);
                ret += self.gamma.powi(current_depth) * model_step.reward;
            }

            ret
        }
    }

    /// Backup phase
    ///
    /// the return generated by the simulated episode is backed up to update, or to
    /// initialize, the action values attached to the edges of the tree traversed by
    /// the tree policy in this iteration of MCTS. No values are saved for the states
    /// and actions visited by the rollout policy beyond the tree.
    fn backup() {
        todo!()
    }

    fn choose_action() -> A {
        todo!()
    }
}

impl<T, P, M, S, A, E> Policy for MCTSPolicy<T, P, M, S, A, E>
where
    T: NodeSelector<State = S, Action = A>,
    P: Policy<State = S, Action = A>,
    M: Model<State = S, Action = A>,
    E: EnvironmentEssay<State = S, Action = A>
        + DiscreteActionEnvironmentEssay<State = S, Action = A>,
{
    type State = S;
    type Action = A;

    fn step<R>(&self, state: S, rng: &mut R) -> Result<A, crate::policy::PolicyError>
    where
        R: rand::Rng + ?Sized,
    {
        /* for i in 0..self.iterations {
            let node = self.selection();
            let ret = self.rollout();
            self.backup();
        }

        let action = self.choose_action(); */
        //Ok(action)
        todo!()
    }

    fn get_best_a(&self, state: S) -> Result<A, crate::policy::PolicyError> {
        todo!()
    }

    fn action_prob(&self, state: S, action: A) -> f32 {
        todo!()
    }
}

/// Action selector in the MCTS tree
///
/// According to the value and the number of visits, the node
/// selector returns the action to take in the current node
pub trait NodeSelector {
    type State;
    type Action;

    /// Returns an action obtained with expansion
    ///
    /// If no expansion is available then it is returned None
    fn expansion_action(
        &self,
        node: Arc<RwLock<arena_tree::Node<NodeAttributes<Self::State, Self::Action>>>>,
    ) -> Result<Option<Self::Action>, MCTSError>;

    fn select_action(
        &self,
        node: Arc<RwLock<arena_tree::Node<NodeAttributes<Self::State, Self::Action>>>>,
    ) -> Result<Self::Action, MCTSError>;
}

pub struct UCBSelector<S, A> {
    exploration: f32,
    marker_s: PhantomData<S>,
    marker_a: PhantomData<A>,
}

impl<S, A> UCBSelector<S, A> {
    fn new(exploration: f32) -> UCBSelector<S, A> {
        UCBSelector {
            exploration,
            marker_s: PhantomData::<S>,
            marker_a: PhantomData::<A>,
        }
    }
}

impl<S, A> NodeSelector for UCBSelector<S, A>
where
    A: Copy,
    S: Copy,
{
    type State = S;
    type Action = A;

    fn expansion_action(
        &self,
        node: Arc<RwLock<arena_tree::Node<NodeAttributes<Self::State, Self::Action>>>>,
    ) -> Result<Option<Self::Action>, MCTSError> {
        let read_guard = node.read().unwrap();
        // First of all, if there is some action that is never visited we return it
        let no_visit_actions: Vec<&A> = read_guard
            .attributes
            .action_visits
            .iter()
            .zip(&read_guard.attributes.actions)
            .filter(|(&x, _)| x == 0)
            .map(|(_, a)| a)
            .collect();
        if !no_visit_actions.is_empty() {
            Ok(Some(*no_visit_actions[0]))
        } else {
            Ok(None)
        }
    }

    fn select_action(
        &self,
        node: Arc<RwLock<arena_tree::Node<NodeAttributes<Self::State, Self::Action>>>>,
    ) -> Result<Self::Action, MCTSError> {
        let read_guard = node.read().unwrap();

        // compute bounds and return the action with the highest bound
        let bounds = read_guard.attributes.action_values.clone()
            + self.exploration
                * ((read_guard.attributes.visits as f32).log10()
                    / node
                        .read()
                        .unwrap()
                        .attributes
                        .action_visits
                        .mapv(|x| x as f32))
                .mapv(|x| x.sqrt());
        println!("{:?}", bounds);
        let idx = bounds.argmax().unwrap();
        Ok(read_guard.attributes.actions[idx] as Self::Action)
    }
}

#[derive(thiserror::Error)]
pub enum MCTSError {
    #[error("Failed to select node")]
    NodeSelection,

    #[error("Failed to compute action")]
    GenericError,
}

impl Debug for MCTSError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self)?;
        if let Some(source) = self.source() {
            writeln!(f, "Caused by:\n\t{}", source)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::{sync::Arc, vec};

    use ndarray::Array1;

    use crate::utils::arena_tree::{self, TreeArena};

    use super::{NodeAttributes, NodeSelector, UCBSelector};

    #[test]
    fn test_ucb_selector() {
        let selector = UCBSelector::<i32, i32>::new(1.0);

        let arena: TreeArena<NodeAttributes<i32, i32>> = arena_tree::TreeArena::new();

        let _ = arena
            .add_node(
                None,
                NodeAttributes::new(0, vec![0, 1, 2], false, None, 1.0),
            )
            .unwrap();

        let node = arena.get_root().unwrap();

        // Check intial exploration, if the visit counts are equal to 0 then the selector
        // choses the first action without visits
        assert_eq!(selector.select_action(Arc::clone(&node)).unwrap(), 0);

        // change action 0 visit count
        node.write().unwrap().attributes.action_visits[0] = 1;
        assert_eq!(selector.select_action(Arc::clone(&node)).unwrap(), 1);

        node.write().unwrap().attributes.action_visits[1] = 1;
        assert_eq!(selector.select_action(Arc::clone(&node)).unwrap(), 2);

        node.write().unwrap().attributes.action_visits[2] = 1;

        // Now we update the values and check that the selector takes the action with highest bound
        node.write().unwrap().attributes.action_values = Array1::from(vec![1.0, 0.0, 0.0]);
        let tot_visits = node.read().unwrap().attributes.action_visits.sum();
        node.write().unwrap().attributes.visits = tot_visits;
        assert_eq!(selector.select_action(Arc::clone(&node)).unwrap(), 0);

        node.write().unwrap().attributes.action_values = Array1::from(vec![2.0, 1.0, 0.0]);
        node.write().unwrap().attributes.action_visits = Array1::from(vec![500, 1, 4]);
        let tot_visits = node.read().unwrap().attributes.action_visits.sum();
        node.write().unwrap().attributes.visits = tot_visits;
        assert_eq!(selector.select_action(Arc::clone(&node)).unwrap(), 1);
    }
}
