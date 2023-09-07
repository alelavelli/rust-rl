use std::{
    cell::RefCell,
    error::Error,
    fmt::{Debug, Display},
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use indicatif::{ProgressBar, ProgressIterator};
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
    parent_source_action_id: Option<usize>,
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
        parent_source_action_id: Option<usize>,
    ) -> NodeAttributes<S, A> {
        let n_actions = actions.len();
        NodeAttributes {
            state,
            actions,
            terminal,
            parent_source_action,
            parent_source_action_id,
            visits: 1,
            value: 0.0,
            action_visits: Array1::zeros(n_actions),
            action_values: Array1::zeros(n_actions),
        }
    }
}

pub enum RootActionCriterion {
    MostPlayed,
    HighestScore,
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
    S: Clone + Display + Debug,
    A: Display + Debug,
{
    tree_policy: T,
    rollout_policy: P,
    model: M,
    iterations: i32,
    max_depth: i32,
    root_action_criterion: RootActionCriterion,
    reuse_tree: bool,
    verbosity: VerbosityConfig,
    tree: RefCell<TreeArena<NodeAttributes<S, A>>>,
    env_essay: E,
    gamma: f32,
}

impl<T, P, M, S, A, E> MCTSPolicy<T, P, M, S, A, E>
where
    T: NodeSelector<State = S, Action = A>,
    P: Policy<State = S, Action = A>,
    M: Model<State = S, Action = A>,
    E: EnvironmentEssay<State = S, Action = A>
        + DiscreteActionEnvironmentEssay<State = S, Action = A>,
    S: Clone + Display + Debug,
    A: Clone + Display + Debug,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tree_policy: T,
        rollout_policy: P,
        model: M,
        iterations: i32,
        max_depth: i32,
        root_action_criterion: RootActionCriterion,
        reuse_tree: bool,
        verbosity: VerbosityConfig,
        env_essay: E,
        gamma: f32,
    ) -> MCTSPolicy<T, P, M, S, A, E> {
        MCTSPolicy {
            tree_policy,
            rollout_policy,
            model,
            iterations,
            max_depth,
            root_action_criterion,
            reuse_tree,
            verbosity,
            tree: RefCell::new(arena_tree::TreeArena::<NodeAttributes<S, A>>::new()),
            env_essay,
            gamma,
        }
    }

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
            let expansion_action = self
                .tree_policy
                .expansion_action(Arc::clone(&current))
                .unwrap();

            /*
            to go on, we need three variables:
                - action: the selected action
                - action_id: the parent's id for the action
                - actual_expansion: if the parent used expansion or no

            If there is no expansion then a already played action is chosen
            */
            let (action, action_id, actual_expansion) = {
                if let Some((action, action_id)) = expansion_action {
                    (action, action_id, true)
                } else {
                    let tree_policy_action = self
                        .tree_policy
                        .select_action(Arc::clone(&current))
                        .unwrap();
                    (tree_policy_action.0, tree_policy_action.1, false)
                }
            };

            // After the aciton is selected then we make a step with the model
            let model_step = self
                .model
                .predict_step(&current.read().unwrap().attributes.state, &action);

            // todo!("You need to handle the selfloop in a state and the repetition of the same state in the tree");

            // if there is an expansion then we add the new tree to the arena as a child of node
            if actual_expansion {
                let available_action = self.env_essay.available_actions(&model_step.state);
                let is_terminal = self.env_essay.is_terminal(&model_step.state);
                let current_node_id = { current.read().unwrap().id };
                let new_node_id = self
                    .tree
                    .borrow()
                    .add_node(
                        Some(current_node_id),
                        NodeAttributes::new(
                            model_step.state,
                            available_action,
                            is_terminal,
                            Some(action),
                            Some(action_id),
                        ),
                    )
                    .unwrap();

                current = self.tree.borrow().get_node(new_node_id).unwrap();
                break;
            } else {
                // otherwise, the next current node is the child of the node with index
                // equal to the action id
                let next_node = Arc::clone(
                    &self
                        .tree
                        .borrow()
                        .get_children(current.read().unwrap().id)
                        .unwrap()[action_id],
                );
                current = next_node;
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
    fn rollout<R>(
        &self,
        node: Arc<RwLock<arena_tree::Node<NodeAttributes<S, A>>>>,
        rng: &mut R,
    ) -> f32
    where
        R: Rng + ?Sized,
    {
        // if the node is a terminal one there is nothing to do, so we return 0.0 as return
        if node.read().unwrap().attributes.terminal {
            0.0
        } else {
            let (mut current_state, mut is_terminal, mut current_depth) = {
                let read_guard = node.read().unwrap();
                (
                    read_guard.attributes.state.clone(),
                    read_guard.attributes.terminal,
                    read_guard.depth,
                )
            };

            let mut ret = 0.0;
            while (current_depth < self.max_depth) & !is_terminal {
                let action = self.rollout_policy.step(&current_state, rng).unwrap();
                let model_step = self.model.predict_step(&current_state, &action);
                current_depth += 1;
                current_state = model_step.state.clone();
                is_terminal = self.env_essay.is_terminal(&current_state);
                ret += self.gamma.powi(current_depth) * model_step.reward;
            }
            // here a value model can be used to get the value of the last state.
            // This depends mainly on the size of the discount factor that determines
            // if this value is significative or is near to zero
            // an example should be
            // `ret += self.value_function(&current_state)`
            ret
        }
    }

    /// Backup phase
    ///
    /// the return generated by the simulated episode is backed up to update, or to
    /// initialize, the action values attached to the edges of the tree traversed by
    /// the tree policy in this iteration of MCTS. No values are saved for the states
    /// and actions visited by the rollout policy beyond the tree.
    fn backup(&self, leaf: Arc<RwLock<arena_tree::Node<NodeAttributes<S, A>>>>, ret: f32) {
        // the value of the leaf is equal to the return of the rollout
        {
            leaf.write().unwrap().attributes.value = ret;
        }

        /*
        then we go up to each parent and we update their value
        here, we use two variables:
            - parent: the parent node to be updated
            - child: the child node that has the updated value
        */

        let mut parent_id = { leaf.read().unwrap().parent };
        let mut child = leaf;

        while parent_id.is_some() {
            {
                let parent = self.tree.borrow().get_node(parent_id.unwrap()).unwrap();
                let mut parent_write_guard = parent.write().unwrap();

                parent_write_guard.attributes.visits += 1;
                parent_write_guard.attributes.action_visits[child
                    .read()
                    .unwrap()
                    .attributes
                    .parent_source_action_id
                    .unwrap()] += 1;
                parent_write_guard.attributes.action_values[child
                    .read()
                    .unwrap()
                    .attributes
                    .parent_source_action_id
                    .unwrap()] = {
                    self.env_essay.compute_reward(
                        &parent_write_guard.attributes.state,
                        child
                            .read()
                            .unwrap()
                            .attributes
                            .parent_source_action
                            .as_ref()
                            .unwrap(),
                        &child.read().unwrap().attributes.state,
                    ) + self.gamma.powi(parent_write_guard.depth + 1)
                        * child.read().unwrap().attributes.value
                };
            }

            // get ready for a new loop. The parent become the child
            child = self.tree.borrow().get_node(parent_id.unwrap()).unwrap();
            parent_id = child.read().unwrap().parent;
        }
    }

    fn choose_action(&self, root: Arc<RwLock<arena_tree::Node<NodeAttributes<S, A>>>>) -> A {
        let chose_action_id = match self.root_action_criterion {
            RootActionCriterion::HighestScore => root
                .read()
                .unwrap()
                .attributes
                .action_values
                .argmax()
                .unwrap(),
            RootActionCriterion::MostPlayed => root
                .read()
                .unwrap()
                .attributes
                .action_visits
                .argmax()
                .unwrap(),
        };
        root.read().unwrap().attributes.actions[chose_action_id].clone()
    }
}

impl<T, P, M, S, A, E> Policy for MCTSPolicy<T, P, M, S, A, E>
where
    T: NodeSelector<State = S, Action = A>,
    P: Policy<State = S, Action = A>,
    M: Model<State = S, Action = A>,
    E: EnvironmentEssay<State = S, Action = A>
        + DiscreteActionEnvironmentEssay<State = S, Action = A>,
    S: Clone + Display + Debug,
    A: Clone + Display + Debug,
{
    type State = S;
    type Action = A;

    fn step<R>(&self, state: &S, rng: &mut R) -> Result<A, crate::policy::PolicyError>
    where
        R: rand::Rng + ?Sized,
    {
        let root = {
            let tree_ref = self.tree.borrow();
            tree_ref.reset();
            let root_id = tree_ref
                .add_node(
                    None,
                    NodeAttributes::new(
                        state.clone(),
                        self.env_essay.available_actions(state),
                        self.env_essay.is_terminal(state),
                        None,
                        None,
                    ),
                )
                .unwrap();
            tree_ref.get_node(root_id).unwrap()
        };

        let progress_bar = if self.verbosity.episode_progress {
            ProgressBar::new(self.iterations as u64)
        }
        else {
            ProgressBar::hidden()
        };
        for _i in (0..self.iterations).progress_with(progress_bar) {
            let node = self.selection(Arc::clone(&root));
            let ret = self.rollout(Arc::clone(&node), rng);
            self.backup(Arc::clone(&node), ret);
        }

        let action = self.choose_action(Arc::clone(&root));
        Ok(action)
    }

    fn get_best_a(&self, _state: &S) -> Result<A, crate::policy::PolicyError> {
        todo!()
    }

    fn action_prob(&self, _state: &S, _action: &A) -> f32 {
        todo!()
    }
}

type NodeType<S, A> = Arc<RwLock<arena_tree::Node<NodeAttributes<S, A>>>>;

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
        node: NodeType<Self::State, Self::Action>,
    ) -> Result<Option<(Self::Action, usize)>, MCTSError>;

    /// Retrurns an action with selection
    fn select_action(
        &self,
        node: NodeType<Self::State, Self::Action>,
    ) -> Result<(Self::Action, usize), MCTSError>;
}

pub struct UCBSelector<S, A> {
    exploration: f32,
    marker_s: PhantomData<S>,
    marker_a: PhantomData<A>,
}

impl<S, A> UCBSelector<S, A> {
    pub fn new(exploration: f32) -> UCBSelector<S, A> {
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
        node: NodeType<Self::State, Self::Action>,
    ) -> Result<Option<(Self::Action, usize)>, MCTSError> {
        let read_guard = node.read().unwrap();
        // First of all, if there is some action that is never visited we return it
        let no_visit_actions: Vec<(&A, usize)> = read_guard
            .attributes
            .action_visits
            .iter()
            .zip(&read_guard.attributes.actions)
            .enumerate()
            .filter(|(_i, (&x, _))| x == 0)
            .map(|(i, (_, a))| (a, i))
            .collect();
        if !no_visit_actions.is_empty() {
            Ok(Some((*no_visit_actions[0].0, no_visit_actions[0].1)))
        } else {
            Ok(None)
        }
    }

    fn select_action(
        &self,
        node: NodeType<Self::State, Self::Action>,
    ) -> Result<(Self::Action, usize), MCTSError> {
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
        let idx = bounds.argmax().unwrap();
        Ok((read_guard.attributes.actions[idx] as Self::Action, idx))
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
                NodeAttributes::new(0, vec![0, 1, 2], false, None, None),
            )
            .unwrap();

        let node = arena.get_root().unwrap();

        // Check intial exploration, if the visit counts are equal to 0 then the selector
        // choses the first action without visits
        assert_eq!(
            selector.expansion_action(Arc::clone(&node)).unwrap(),
            Some((0, 0))
        );

        // change action 0 visit count
        node.write().unwrap().attributes.action_visits[0] = 1;
        assert_eq!(
            selector.expansion_action(Arc::clone(&node)).unwrap(),
            Some((1, 1))
        );

        node.write().unwrap().attributes.action_visits[1] = 1;
        assert_eq!(
            selector.expansion_action(Arc::clone(&node)).unwrap(),
            Some((2, 2))
        );

        node.write().unwrap().attributes.action_visits[2] = 1;

        // Now we update the values and check that the selector takes the action with highest bound
        node.write().unwrap().attributes.action_values = Array1::from(vec![1.0, 0.0, 0.0]);
        let tot_visits = node.read().unwrap().attributes.action_visits.sum();
        node.write().unwrap().attributes.visits = tot_visits;
        assert_eq!(selector.select_action(Arc::clone(&node)).unwrap(), (0, 0));

        node.write().unwrap().attributes.action_values = Array1::from(vec![2.0, 1.0, 0.0]);
        node.write().unwrap().attributes.action_visits = Array1::from(vec![500, 1, 4]);
        let tot_visits = node.read().unwrap().attributes.action_visits.sum();
        node.write().unwrap().attributes.visits = tot_visits;
        assert_eq!(selector.select_action(Arc::clone(&node)).unwrap(), (1, 1));
    }
}
