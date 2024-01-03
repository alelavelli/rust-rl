use ndarray::{Array2, ArrayBase, Dim, ViewRepr};

use std::fmt::Debug;

pub mod normalization;
pub mod ohe;
pub mod polynomial;

#[derive(thiserror::Error, Debug)]
pub enum PreprocessingError {
    #[error("Failed to fit data")]
    FitError,

    #[error("Failed to transform data")]
    TransformError,
}

pub trait Preprocessor<T> {
    fn fit(
        &mut self,
        x: &ArrayBase<ViewRepr<&T>, Dim<[usize; 2]>>,
    ) -> Result<(), PreprocessingError>;

    fn transform(
        &self,
        x: &ArrayBase<ViewRepr<&T>, Dim<[usize; 2]>>,
    ) -> Result<Array2<f32>, PreprocessingError>;

    fn inverse_transform(
        &self,
        x: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<Array2<T>, PreprocessingError>;
}
