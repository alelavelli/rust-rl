use crate::{learn::VerbosityConfig, model::Model, policy::Policy};

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
/// - `ucb_max_value`: the maximum possibile value used for computing ucbs
struct NodeAttributes {
    state: i32,
    actions: i32,
    terminal: bool,
    parent_source_action: i32,
    visits: i32,
    value: f32,
    action_visits: Vec<i32>,
    action_values: Vec<f32>,
    ucb_max_value: f32,
}

/// Parameters structure for MCTS algorithm
///
/// - `tree_policy`: policy that chooses actions inside the tree
/// - `rollout_policy`: policy used during the rollout phase
/// - `model`: transition model used during rollout
/// - `iterations`: number of iterations before returning the chosen action
/// - `max_horizon`: the maximum horizon to simulate forward steps
///     (i.e., the maximum depth of the tree)
/// - `root_action_criterion`: choose the most played action at the root
///     or the one with highest score. It is an Enum
/// - `scale_values`: whether to rescale the values of each node by their maximum value
/// - `reuse_tree`: whether to start MCTS from the old sub-tree whenever the new state is close to its old prediction
/// - `reuse_thresh`: threshold for deciding when to reuse the old tree (l2-norm deviation between new/old state)
///
pub struct Params {
    tree_policy: i32,
    rollout_policy: i32,
    model: i32,
    iterations: i32,
    max_horizon: i32,
    root_action_criterion: i32,
    scale_values: i32,
    reuse_tree: i32,
    reuse_thresh: i32,
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
/// - `max_horizon`: the maximum horizon to simulate forward steps
///     (i.e., the maximum depth of the tree)
/// - `root_action_criterion`: choose the most played action at the root
///     or the one with highest score. It is an Enum
/// - `scale_values`: whether to rescale the values of each node by their maximum value
/// - `reuse_tree`: whether to start MCTS from the old sub-tree whenever the new state is close to its old prediction
/// - `reuse_threshold`: threshold for deciding when to reuse the old tree (l2-norm deviation between new/old state)
/// - `verbosity`: verbosity configuration struct
pub struct MCTSPolicy<T, P, M, S, A>
where
    T: Policy<State = S, Action = A>,
    P: Policy<State = S, Action = A>,
    M: Model<State = S, Action = A>,
{
    tree_policy: T,
    rollout_policy: P,
    model: M,
    iterations: i32,
    max_horizon: i32,
    root_action_criterion: RootActionCriterion,
    scale_value: bool,
    reuse_tree: bool,
    reuse_threshold: f32,
    verbosity: VerbosityConfig,
}

impl<T, P, M, S, A> Policy for MCTSPolicy<T, P, M, S, A>
where
    T: Policy<State = S, Action = A>,
    P: Policy<State = S, Action = A>,
    M: Model<State = S, Action = A>,
{
    type State = S;
    type Action = A;

    fn step<R>(&self, state: S, rng: &mut R) -> Result<A, crate::policy::PolicyError>
    where
        R: rand::Rng + ?Sized,
    {
        todo!()
    }

    fn get_best_a(&self, state: S) -> Result<A, crate::policy::PolicyError> {
        todo!()
    }

    fn action_prob(&self, state: S, action: A) -> f32 {
        todo!()
    }
}
