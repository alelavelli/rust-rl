use std::{sync::{RwLock, Arc}, collections::HashMap};

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
    id: NodeId,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    // depth of the node inside the tree. 0 if it is root
    depth: i32,
    attributes: T,
}

/// TreeArena struct that owns the nodes
pub struct TreeArena<T> {
    map: RwLock<HashMap<NodeId, Arc<RwLock<Node<T>>>>>,
    global_counter: RwLock<NodeId>,
    root: Option<NodeId>
}

impl<T> TreeArena<T> {
    pub fn new() -> TreeArena<T> {
        TreeArena { map: RwLock::new(HashMap::new()), global_counter: RwLock::new(0) }
    }

    /// Generate new id in the arena and delete the previous one
    fn generate_id(&self) -> NodeId {
        *self.global_counter.write().unwrap() += 1;
        self.global_counter.read().unwrap().clone()
    }

    /// Return node if it exists in the arena
    pub fn get_node(&self, node_id: NodeId) -> Option<Arc<RwLock<Node<T>>>> {
        self.map.read().unwrap().get(&node_id).map(Arc::clone)
    }

    /// Add a node in the arena with the given parent
    pub fn add_node(&self, parent: Option<NodeId>, attributes: T) -> Result<NodeId, TreeError> {
        if parent.is_none() & self.root.is_some() {
            Err(TreeError::RootAlreadyExists)

        } else {

            let new_id = self.generate_id();

            let node = Node {
                id: new_id,
                parent: parent,
                children: Vec::new(),
                depth: if let Some(parent_id) = parent { self.get_node(parent_id).unwrap().read().unwrap().id } else { 0 },
                attributes
            };

            {
                self.map.write().unwrap().insert(new_id, Arc::new(RwLock::new(node)));
            } // unlock here the map

            Ok(new_id)
        }
    }

    /// Remove the node and delete its subtree
    /// 
    /// If the node does not exist then it returs Ok since it's not a problem
    pub fn remove_node(&self, node_id: NodeId) -> Result<(), TreeError> {
        if self.get_node(node_id).is_some() {
            // we wrap it in a block to delete the write lock guard
            {
                self.map.write().unwrap().remove(&node_id).unwrap()
            };
            // it could be happen that the read block will block the write lock in the next the remove_node 
            for c in self.get_children(node_id).unwrap() {
                let result = self.remove_node(c.read().unwrap().id);
                if result.is_err() {
                    return Err(TreeError::NodeRemoveError)
                }
            }
        }
        Ok(())
    }

    /// Check if the node is in the arena
    pub fn has_node(&self, node_id: NodeId) -> bool {
        self.map.read().unwrap().contains_key(&node_id)
    }

    /// Return the root Node of the arena
    pub fn get_root(&self) -> Option<Arc<RwLock<Node<T>>>> {
        self.root.map(|node_id| self.get_node(node_id).unwrap())
    }

    /// Return the vector of children of the given node. Err if the node does not exists
    pub fn get_children(&self, parent: NodeId) -> Result<Vec<Arc<RwLock<Node<T>>>>, TreeError> {
        if let Some(node) = self.get_node(parent) {
            Ok(node.read().unwrap().children.iter().map(|c| Arc::clone(&self.get_node(*c).unwrap())).collect())
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
            Ok(node.read().unwrap().parent.map(|v| self.get_node(v).unwrap()))
        } else {
            Err(TreeError::NodeDoesNotExist)
        }
    }
}

/// We create a new implementation block because this method require that type T has
/// `Clone` Trait implemented
impl<T> TreeArena<T>
where 
    T: Clone
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
        for (other_node_id, other_node_ref) in other.map.read().unwrap().iter() {
            let new_id = self.generate_id();
            id_mapping.insert(*other_node_id, new_id);
        }
        for (other_node_id, other_node_ref) in other.map.read().unwrap().iter() {
            let read_guard = other_node_ref.read().unwrap();
            let new_id = *id_mapping.get(other_node_id).unwrap();
            let new_node = Node {
                attributes: read_guard.attributes.clone(),
                id: new_id,
                parent: if read_guard.parent.is_none() { Some(node_id) } else { Some(*id_mapping.get(&read_guard.parent.unwrap()).unwrap()) },
                children: read_guard.children.iter().map(|x| *id_mapping.get(x).unwrap() ).collect_vec(),
                depth: read_guard.depth + node_depth
            };
            self.map.write().unwrap().insert(new_id, Arc::new(RwLock::new(new_node)));
        }

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
            let parent_id = new_arena.add_node(None, root.read().unwrap().attributes.clone()).unwrap();

            // Add the children
            for child in self.get_children(node_id).unwrap() {
                // first we extract the subtree of the child
                let child_arena = self.extract_subtree(child.read().unwrap().id).unwrap();
                // do the grift of the child arena to the new_arena. The grift function takes the child subtree and add it as
                // a new child of parent_id. The ids of the child subtree will be updated according to the id of new_arena
                // we just take the hashmap of new_arena and put is inside the new_arena map
                // when we do this we need to update the index by an offset equal to the max index the new_arena has
                // then, the depth need also to be update adding the depth of the parent + 1
                new_arena.grift(child_arena, parent_id);
            }
            Ok(new_arena)
        } else {
            Err(TreeError::NodeDoesNotExist)
        }
    }
}