use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use itertools::Itertools;

use super::TreeError;

/// Type alias that represent the nodeId
pub type NodeId = i32;

/// Node struct
///
/// It has an id that is registered in the arena, parent and children
/// are NodeId as well
///
/// The attributes contains actual attributes of the node as is implemented
/// as generic.
pub struct Node<T> {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    // depth of the node inside the tree. 0 if it is root
    pub depth: i32,
    pub attributes: T,
}

/// TreeArena struct that owns the nodes
pub struct TreeArena<T> {
    map: RwLock<HashMap<NodeId, Arc<RwLock<Node<T>>>>>,
    global_counter: RwLock<NodeId>,
    root: RwLock<Option<NodeId>>,
}

impl<T> Default for TreeArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TreeArena<T> {
    pub fn new() -> TreeArena<T> {
        TreeArena {
            map: RwLock::new(HashMap::new()),
            global_counter: RwLock::new(0),
            root: RwLock::new(None),
        }
    }

    // delete all the nodes in the arena tree
    pub fn reset(&self) {
        *self.map.write().unwrap() = HashMap::new();
        *self.global_counter.write().unwrap() = 0;
        *self.root.write().unwrap() = None;
    }

    /// Generate new id in the arena and delete the previous one
    fn generate_id(&self) -> NodeId {
        *self.global_counter.write().unwrap() += 1;
        *self.global_counter.read().unwrap()
    }

    /// Retruns the list of the nodes in the tree
    pub fn node_list(&self) -> Vec<NodeId> {
        self.map.read().unwrap().keys().copied().collect()
    }

    /// Return node if it exists in the arena
    pub fn get_node(&self, node_id: NodeId) -> Option<Arc<RwLock<Node<T>>>> {
        self.map.read().unwrap().get(&node_id).map(Arc::clone)
    }

    /// Add a node in the arena with the given parent
    pub fn add_node(&self, parent: Option<NodeId>, attributes: T) -> Result<NodeId, TreeError> {
        let root_is_present = { self.root.read().unwrap().is_some() };

        if parent.is_none() & root_is_present {
            Err(TreeError::RootAlreadyExists)
        } else {
            // first create the new node id
            // then create the node
            // add it to the map
            // If it is not root node, modify the parent adding it as its child

            let new_id = self.generate_id();

            let node = Node {
                id: new_id,
                parent,
                children: Vec::new(),
                depth: if let Some(parent_id) = parent {
                    self.get_node(parent_id).unwrap().read().unwrap().depth + 1
                } else {
                    0
                },
                attributes,
            };

            {
                self.map
                    .write()
                    .unwrap()
                    .insert(new_id, Arc::new(RwLock::new(node)));
            } // unlock here the map

            if let Some(parent_node) = parent {
                let parent_node = self.get_node(parent_node);
                parent_node.unwrap().write().unwrap().children.push(new_id);
            }

            if !root_is_present {
                // if the arena has no root then we set this new node as the root
                *self.root.write().unwrap() = Some(new_id);
            }
            Ok(new_id)
        }
    }

    /// Remove the node and delete its subtree
    ///
    /// If the node does not exist then it returs Ok since it's not a problem
    pub fn remove_node(&self, node_id: NodeId) -> Result<(), TreeError> {
        if let Some(node) = self.get_node(node_id) {
            // it could be happen that the read block will block the write lock in the next the remove_node
            for c in self.get_children(node_id).unwrap() {
                let result = self.remove_node(c.read().unwrap().id);
                if result.is_err() {
                    return Err(TreeError::NodeRemoveError);
                }
            }

            // If it is the root node then we empty the struct attribute
            if node.read().unwrap().parent.is_none() {
                *self.root.write().unwrap() = None;
            }

            // we wrap it in a block to delete the write lock guard
            self.map.write().unwrap().remove(&node_id).unwrap();

            Ok(())
        } else {
            Err(TreeError::NodeDoesNotExist)
        }
    }

    /// Check if the node is in the arena
    pub fn has_node(&self, node_id: NodeId) -> bool {
        self.map.read().unwrap().contains_key(&node_id)
    }

    /// Return the root Node of the arena
    pub fn get_root(&self) -> Option<Arc<RwLock<Node<T>>>> {
        self.root
            .read()
            .unwrap()
            .map(|node_id| self.get_node(node_id).unwrap())
    }

    /// Return the vector of children of the given node. Err if the node does not exists
    pub fn get_children(&self, parent: NodeId) -> Result<Vec<Arc<RwLock<Node<T>>>>, TreeError> {
        if let Some(node) = self.get_node(parent) {
            Ok(node
                .read()
                .unwrap()
                .children
                .iter()
                .map(|c| Arc::clone(&self.get_node(*c).unwrap()))
                .collect())
        } else {
            Err(TreeError::NodeDoesNotExist)
        }
    }

    /// Returns parent of the node
    ///
    /// - `None` is returned if the parent is missing, i.e., the node is a root
    /// - `Err` is returned if the node does not exist in the arena
    pub fn get_parent(&self, node_id: NodeId) -> Result<Option<Arc<RwLock<Node<T>>>>, TreeError> {
        if let Some(node) = self.get_node(node_id) {
            Ok(node
                .read()
                .unwrap()
                .parent
                .map(|v| self.get_node(v).unwrap()))
        } else {
            Err(TreeError::NodeDoesNotExist)
        }
    }
}

/// We create a new implementation block because this method require that type T has
/// `Clone` Trait implemented
impl<T> TreeArena<T>
where
    T: Clone,
{
    /// Grift the arena into the current one. The root node of the `other` arena is
    /// set as child of `node_id`.
    ///
    /// Value that will change are depth of the `other` arena's nodes and their ids.
    ///
    /// In particular, the ids are created starting from the current available id of
    /// the `self` arena. The depth is increased by the depth of `node_id`
    fn grift(&self, other: TreeArena<T>, node_id: NodeId) -> Result<(), TreeError> {
        let node_depth = self.get_node(node_id).unwrap().read().unwrap().depth;

        // first we create a id mapping that translate the `other` id into self id
        let mut id_mapping: HashMap<NodeId, NodeId> = HashMap::new();
        for (other_node_id, _) in other.map.read().unwrap().iter().sorted_by_key(|a| a.0) {
            let new_id = self.generate_id();
            id_mapping.insert(*other_node_id, new_id);
        }
        for (other_node_id, other_node_ref) in other.map.read().unwrap().iter() {
            let read_guard = other_node_ref.read().unwrap();
            let new_id = *id_mapping.get(other_node_id).unwrap();
            let new_node = Node {
                attributes: read_guard.attributes.clone(),
                id: new_id,
                parent: if read_guard.parent.is_none() {
                    Some(node_id)
                } else {
                    Some(*id_mapping.get(&read_guard.parent.unwrap()).unwrap())
                },
                children: read_guard
                    .children
                    .iter()
                    .map(|x| *id_mapping.get(x).unwrap())
                    .collect_vec(),
                depth: read_guard.depth + node_depth,
            };
            self.map
                .write()
                .unwrap()
                .insert(new_id, Arc::new(RwLock::new(new_node)));
        }
        // add the root of `other` as child of node_id
        self.get_node(node_id)
            .unwrap()
            .write()
            .unwrap()
            .children
            .push(
                *id_mapping
                    .get(&other.get_root().unwrap().read().unwrap().id)
                    .unwrap(),
            );

        Ok(())
    }

    /// Extract subtree starting from the node
    ///
    ///
    /// The extraction is done creating a new arena with root starting from
    /// node. The original arena is not modified, therefore, the nodes will be cloned
    pub fn extract_subtree(&self, node_id: NodeId) -> Result<TreeArena<T>, TreeError> {
        if let Some(root) = self.get_node(node_id) {
            let new_arena = TreeArena::<T>::new();

            // Add the root
            let parent_id = new_arena
                .add_node(None, root.read().unwrap().attributes.clone())
                .unwrap();

            // Add the children
            for child in self.get_children(node_id).unwrap() {
                // first we extract the subtree of the child
                let child_arena = self.extract_subtree(child.read().unwrap().id).unwrap();
                // do the grift of the child arena to the new_arena. The grift function takes the child subtree and add it as
                // a new child of parent_id. The ids of the child subtree will be updated according to the id of new_arena
                // we just take the hashmap of new_arena and put is inside the new_arena map
                // when we do this we need to update the index by an offset equal to the max index the new_arena has
                // then, the depth need also to be update adding the depth of the parent + 1
                new_arena.grift(child_arena, parent_id).unwrap();
            }
            Ok(new_arena)
        } else {
            Err(TreeError::NodeDoesNotExist)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread, time::Duration};

    use itertools::Itertools;

    use crate::utils::{arena_tree::NodeId, TreeError};

    use super::TreeArena;

    struct TestArena {
        arena: TreeArena<i32>,
        root_id: NodeId,
        first_child: NodeId,
        second_child: NodeId,
        first_grandson: NodeId,
        second_grandson: NodeId,
    }

    fn init_test() -> TestArena {
        let arena = TreeArena::<i32>::new();

        assert!(arena.get_root().is_none());

        // Add the root node
        let res = arena.add_node(None, 1);
        assert!(res.is_ok());
        let root_id = res.unwrap();

        // Try to add a new root node
        let res = arena.add_node(None, 10);
        assert!(res.is_err());

        // Add two children to the root node
        let res = arena.add_node(Some(root_id), 10);
        assert!(res.is_ok());
        let first_child = res.unwrap();
        let res = arena.add_node(Some(root_id), 11);
        assert!(res.is_ok());
        let second_child = res.unwrap();

        // Add two children to the second child
        let res = arena.add_node(Some(second_child), 100);
        assert!(res.is_ok());
        let first_grandson = res.unwrap();
        let res = arena.add_node(Some(second_child), 101);
        assert!(res.is_ok());
        let second_grandson = res.unwrap();

        TestArena {
            arena,
            root_id,
            first_child,
            second_child,
            first_grandson,
            second_grandson,
        }
    }

    #[test]
    fn test_basic_arena() {
        let test_arena = init_test();

        // test get_root
        assert_eq!(
            test_arena.root_id,
            test_arena.arena.get_root().unwrap().read().unwrap().id
        );

        // test get_children
        assert_eq!(
            vec![test_arena.first_child, test_arena.second_child],
            test_arena
                .arena
                .get_children(test_arena.root_id)
                .unwrap()
                .iter()
                .map(|x| x.read().unwrap().id)
                .collect_vec()
        );
        assert_eq!(
            Vec::<NodeId>::new(),
            test_arena
                .arena
                .get_children(test_arena.first_child)
                .unwrap()
                .iter()
                .map(|x| x.read().unwrap().id)
                .collect_vec()
        );
        assert_eq!(
            vec![test_arena.first_grandson, test_arena.second_grandson],
            test_arena
                .arena
                .get_children(test_arena.second_child)
                .unwrap()
                .iter()
                .map(|x| x.read().unwrap().id)
                .collect_vec()
        );

        // test get_children of missing node
        assert_eq!(
            TreeError::NodeDoesNotExist,
            test_arena.arena.get_children(40).err().unwrap()
        );

        // test node_list
        let mut node_list = test_arena.arena.node_list();
        node_list.sort();
        assert_eq!(
            vec![
                test_arena.root_id,
                test_arena.first_child,
                test_arena.second_child,
                test_arena.first_grandson,
                test_arena.second_grandson
            ],
            node_list
        );

        // test get_parent
        assert_eq!(
            test_arena.root_id,
            test_arena
                .arena
                .get_parent(test_arena.first_child)
                .unwrap()
                .unwrap()
                .read()
                .unwrap()
                .id
        );
        assert_eq!(
            test_arena.root_id,
            test_arena
                .arena
                .get_parent(test_arena.second_child)
                .unwrap()
                .unwrap()
                .read()
                .unwrap()
                .id
        );
        assert_eq!(
            test_arena.second_child,
            test_arena
                .arena
                .get_parent(test_arena.first_grandson)
                .unwrap()
                .unwrap()
                .read()
                .unwrap()
                .id
        );
        assert_eq!(
            test_arena.second_child,
            test_arena
                .arena
                .get_parent(test_arena.second_grandson)
                .unwrap()
                .unwrap()
                .read()
                .unwrap()
                .id
        );

        // test get_parent for missing node
        assert_eq!(
            TreeError::NodeDoesNotExist,
            test_arena.arena.get_children(40).err().unwrap()
        );

        // test has_node
        assert!(test_arena.arena.has_node(test_arena.root_id));
        assert!(!test_arena.arena.has_node(40));

        // test get_node
        assert_eq!(
            test_arena.first_child,
            test_arena
                .arena
                .get_node(test_arena.first_child)
                .unwrap()
                .read()
                .unwrap()
                .id
        );

        // test get_node for missing node
        assert!(test_arena.arena.get_node(40).is_none());

        // test depth
        assert_eq!(
            0,
            test_arena
                .arena
                .get_node(test_arena.root_id)
                .unwrap()
                .read()
                .unwrap()
                .depth
        );
        assert_eq!(
            1,
            test_arena
                .arena
                .get_node(test_arena.first_child)
                .unwrap()
                .read()
                .unwrap()
                .depth
        );
        assert_eq!(
            1,
            test_arena
                .arena
                .get_node(test_arena.second_child)
                .unwrap()
                .read()
                .unwrap()
                .depth
        );
        assert_eq!(
            2,
            test_arena
                .arena
                .get_node(test_arena.first_grandson)
                .unwrap()
                .read()
                .unwrap()
                .depth
        );
        assert_eq!(
            2,
            test_arena
                .arena
                .get_node(test_arena.second_grandson)
                .unwrap()
                .read()
                .unwrap()
                .depth
        );
    }

    #[test]
    fn test_remove_leaf() {
        let test_arena = init_test();

        assert_eq!(5, test_arena.arena.node_list().len());
        let res = test_arena.arena.remove_node(test_arena.first_grandson);
        assert!(res.is_ok());
        assert!(test_arena
            .arena
            .get_node(test_arena.first_grandson)
            .is_none());
        assert_eq!(4, test_arena.arena.node_list().len());
    }

    #[test]
    fn test_remove_node() {
        let test_arena = init_test();

        assert_eq!(5, test_arena.arena.node_list().len());
        let res = test_arena.arena.remove_node(test_arena.second_child);
        assert!(res.is_ok());
        assert!(test_arena.arena.get_node(test_arena.second_child).is_none());
        assert!(test_arena
            .arena
            .get_node(test_arena.first_grandson)
            .is_none());
        assert!(test_arena
            .arena
            .get_node(test_arena.second_grandson)
            .is_none());
        assert_eq!(2, test_arena.arena.node_list().len());
    }

    #[test]
    fn test_remove_root() {
        let test_arena = init_test();

        assert_eq!(5, test_arena.arena.node_list().len());
        let res = test_arena.arena.remove_node(test_arena.root_id);
        assert!(res.is_ok());
        assert!(test_arena.arena.get_node(test_arena.root_id).is_none());
        assert!(test_arena.arena.get_node(test_arena.first_child).is_none());
        assert!(test_arena.arena.get_node(test_arena.second_child).is_none());
        assert!(test_arena
            .arena
            .get_node(test_arena.first_grandson)
            .is_none());
        assert!(test_arena
            .arena
            .get_node(test_arena.second_grandson)
            .is_none());
        assert_eq!(0, test_arena.arena.node_list().len());

        // add a new root
        let res = test_arena.arena.add_node(None, 1);
        assert!(res.is_ok());
    }

    #[test]
    fn test_remove_missing() {
        let arena = TreeArena::<i32>::new();

        assert!(arena.get_root().is_none());

        // Add the root node
        let res = arena.add_node(None, 1);
        assert!(res.is_ok());
        let root_id = res.unwrap();

        // Try to add a new root node
        let res = arena.add_node(None, 10);
        assert!(res.is_err());

        // Add two children to the root node
        let res = arena.add_node(Some(root_id), 10);
        assert!(res.is_ok());
        let res = arena.add_node(Some(root_id), 11);
        assert!(res.is_ok());
        let second_child = res.unwrap();

        // Add two children to the second child
        let res = arena.add_node(Some(second_child), 100);
        assert!(res.is_ok());
        let res = arena.add_node(Some(second_child), 101);
        assert!(res.is_ok());

        assert_eq!(5, arena.node_list().len());
        let res = arena.remove_node(45);
        assert!(res.is_err());
        assert_eq!(5, arena.node_list().len());
    }

    #[test]
    fn test_extract_subtree_leaf() {
        let test_arena = init_test();

        let extracted_arena = test_arena.arena.extract_subtree(test_arena.second_grandson);

        assert!(extracted_arena.is_ok());

        let extracted_arena = extracted_arena.unwrap();

        assert_eq!(extracted_arena.node_list(), vec![1]);
        assert_eq!(
            extracted_arena
                .get_node(1)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.second_grandson)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
        );
    }

    #[test]
    fn test_extract_subtree_node() {
        let test_arena = init_test();

        let extracted_arena = test_arena.arena.extract_subtree(test_arena.second_child);

        assert!(extracted_arena.is_ok());

        let extracted_arena = extracted_arena.unwrap();

        let mut node_list = extracted_arena.node_list();
        node_list.sort();

        // verify that there are three nodes
        assert_eq!(node_list, vec![1, 2, 3]);

        // verify the root and nodes
        assert_eq!(
            extracted_arena
                .get_node(1)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.second_child)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
        );
        assert_eq!(
            extracted_arena
                .get_node(2)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.first_grandson)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
        );
        assert_eq!(
            extracted_arena
                .get_node(3)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.second_grandson)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
        );
        assert_eq!(
            extracted_arena
                .get_root()
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.second_child)
                .unwrap()
                .read()
                .unwrap()
                .attributes
        );
        // verify that the parent of the ndoes is the root
        assert_eq!(
            extracted_arena.get_node(2).unwrap().read().unwrap().parent,
            Some(1),
        );
        assert_eq!(
            extracted_arena.get_node(3).unwrap().read().unwrap().parent,
            Some(1)
        );

        // verify the children
        assert_eq!(
            extracted_arena
                .get_node(1)
                .unwrap()
                .read()
                .unwrap()
                .children,
            vec![2, 3]
        );
    }

    #[test]
    fn test_extract_subtree_root() {
        let test_arena = init_test();

        let extracted_arena = test_arena.arena.extract_subtree(test_arena.root_id);

        assert!(extracted_arena.is_ok());

        let extracted_arena = extracted_arena.unwrap();

        let mut node_list = extracted_arena.node_list();
        node_list.sort();

        // verify that there are 5 nodes
        assert_eq!(node_list, vec![1, 2, 3, 4, 5]);

        // verify that the root node is the same
        assert_eq!(
            extracted_arena
                .get_node(1)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.root_id)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
        );

        // verify that the other nodes are the same
        assert_eq!(
            extracted_arena
                .get_node(2)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.first_child)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
        );
        assert_eq!(
            extracted_arena
                .get_node(3)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.second_child)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
        );
        assert_eq!(
            extracted_arena
                .get_node(4)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.first_grandson)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
        );
        assert_eq!(
            extracted_arena
                .get_node(5)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.second_grandson)
                .unwrap()
                .read()
                .unwrap()
                .attributes,
        );

        // verify that the root is actually the older root
        assert_eq!(
            extracted_arena
                .get_root()
                .unwrap()
                .read()
                .unwrap()
                .attributes,
            test_arena
                .arena
                .get_node(test_arena.root_id)
                .unwrap()
                .read()
                .unwrap()
                .attributes
        );

        // verify the parent of first_child and second_child
        assert_eq!(
            extracted_arena.get_node(2).unwrap().read().unwrap().parent,
            Some(1),
        );
        assert_eq!(
            extracted_arena.get_node(3).unwrap().read().unwrap().parent,
            Some(1)
        );

        // verify the parent of first_grandson and second_grandson
        assert_eq!(
            extracted_arena.get_node(4).unwrap().read().unwrap().parent,
            Some(3),
        );
        assert_eq!(
            extracted_arena.get_node(5).unwrap().read().unwrap().parent,
            Some(3)
        );

        // verify children of root
        assert_eq!(
            extracted_arena
                .get_node(1)
                .unwrap()
                .read()
                .unwrap()
                .children,
            vec![2, 3]
        );

        // verify children of node
        assert_eq!(
            extracted_arena
                .get_node(3)
                .unwrap()
                .read()
                .unwrap()
                .children,
            vec![4, 5]
        );
    }

    #[test]
    fn test_extract_subtree_missing() {
        let test_arena = init_test();

        let extracted_arena = test_arena.arena.extract_subtree(43);

        assert!(extracted_arena.is_err());
    }

    #[test]
    fn test_multithread_read() {
        let test_arena = init_test();

        let arc_test_vars = Arc::new(test_arena);
        let arc_test_vars_0 = Arc::clone(&arc_test_vars);
        let arc_test_vars_1 = Arc::clone(&arc_test_vars);

        let handle_0 = thread::spawn(move || {
            let mut node_list = arc_test_vars_0.arena.node_list();
            node_list.sort();
            assert_eq!(
                vec![
                    arc_test_vars_0.root_id,
                    arc_test_vars_0.first_child,
                    arc_test_vars_0.second_child,
                    arc_test_vars_0.first_grandson,
                    arc_test_vars_0.second_grandson
                ],
                node_list
            );
        });
        let handle_1 = thread::spawn(move || {
            let mut node_list = arc_test_vars_1.arena.node_list();
            node_list.sort();
            assert_eq!(
                vec![
                    arc_test_vars_1.root_id,
                    arc_test_vars_1.first_child,
                    arc_test_vars_1.second_child,
                    arc_test_vars_1.first_grandson,
                    arc_test_vars_1.second_grandson
                ],
                node_list
            );
        });
        // if one thread panics then the join().unwrap() will panic as well
        handle_0.join().unwrap();
        handle_1.join().unwrap();
    }

    #[test]
    fn test_multithread_read_write() {
        // here we test two parallel threads, one updates the arena while the second
        // reads from it

        let test_arena = init_test();

        let arc_test_vars = Arc::new(test_arena);
        let arc_test_vars_0 = Arc::clone(&arc_test_vars);
        let arc_test_vars_1 = Arc::clone(&arc_test_vars);

        let handle_0 = thread::spawn(move || {
            assert_eq!(arc_test_vars_0.arena.node_list().len(), 5);
            arc_test_vars_0
                .arena
                .add_node(Some(arc_test_vars_0.first_child), 5)
                .unwrap();
            assert_eq!(arc_test_vars_0.arena.node_list().len(), 6);
        });
        let handle_1 = thread::spawn(move || {
            thread::sleep(Duration::from_secs(1));
            assert_eq!(arc_test_vars_1.arena.node_list().len(), 6);
        });
        // if one thread panics then the join().unwrap() will panic as well
        handle_0.join().unwrap();
        handle_1.join().unwrap();
    }
}
