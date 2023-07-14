//! Monte Carlo Tree Search for Tabular Data
//!
//! Implementation of MCTS for tabular data focusing on high parallelization
//! of the iterations
//!
//! With MCTS there are three possibilities for parallelization:
//!
//! 1. **Leaf parallelization** is the simplest form of parallelization
//! and applies it in the phase 3 of MCTS algorithm. The main thread executes
//! phase 1 (selection) and phase 2 (expansion), then when a leaf is reached
//! multiple parallel montecarlo simulations are run. Finally, the phase 4
//! (back propagation) is done sequentially. This kind of parallelization is
//! useful when the time required to run a single simulation is long.
//!
//! 2. **Root parallelization** consists of building multiple MCTS trees in
//! parallel, with one thread per tree. When, the time isspent, all the root children
//! of the separate MCTS trees are merged. The best move is selected based on summing
//! up the scores.
//!
//! 3. **Tree Parallelization** uses one shared tree from which several simultaneous games
//! are played. Each thread can modify the information contained in the tree. Therefore,
//! mutexes are used to lock from time to time certain parts of the tree to prevent data
//! corruption. Tree parallelization has two methods to improve its performace:
//!     - **mutex location**: based on the location of the mutexes in the tree, there are
//!        two location methods named *global mutex* and *local mutexes*.
//!     - **virtual loss**: a viartual loss is assigned when a node is visided by a thread.
//!         Hence, the value of this node will be decreased. This is used to avoid multiple
//!         threads visit the same portion of the tree.
//!

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Weak, Rc};

/// Tree node built during the iteration of MCTS
#[derive(Debug)]
pub struct Node {
    pub parent: RefCell<Weak<Node>>,
    pub children: RefCell<HashMap<i32, RefCell<Node>>>,
    pub state: i32,
    pub actions: Vec<i32>,
    pub depth: i32,
    // index of the parent's action that lead to this node
    pub parent_action: Option<i32>,
    pub terminal: bool,
    pub visits: i32,
    pub value: f32,
    // number of visits to each available action
    pub action_visits: HashMap<i32, i32>,
    // value of each available action
    pub action_values: HashMap<i32, f32>,
    // maximum value used for computing ucbs
    pub max_value: f32,
}
