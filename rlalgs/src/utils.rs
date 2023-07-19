pub mod tree;
pub mod arena_tree;


use std::{error::Error, fmt::Debug};

#[derive(thiserror::Error, PartialEq)]
pub enum TreeError {
    #[error("Root Node already exists in the arena")]
    RootAlreadyExists,

    #[error("It's not possible to remove the node")]
    NodeRemoveError,

    #[error("The node does not exist")]
    NodeDoesNotExist,

    #[error("Failed to compute tree operation")]
    GenericError,
}

impl Debug for TreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self)?;
        if let Some(source) = self.source() {
            writeln!(f, "Caused by:\n\t{}", source)?;
        }
        Ok(())
    }
}
