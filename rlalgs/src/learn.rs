use std::{error::Error, fmt::Debug};

use crate::EpisodeGenerationError;

pub mod tabular;

#[derive(thiserror::Error)]
pub enum LearningError {
    #[error("Failed to generate episode")]
    EpisodeGeneration(#[source] EpisodeGenerationError),

    #[error("Failed to learn")]
    GenericError,
}

impl Debug for LearningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self)?;
        if let Some(source) = self.source() {
            writeln!(f, "Caused by:\n\t{}", source)?;
        }
        Ok(())
    }
}
