use itertools::Itertools;
use ndarray::{s, Array2, ArrayBase, Dim, ViewRepr};
use rayon::{prelude::*, ThreadPoolBuilder};

use std::{ops::Mul, sync::RwLock};

use super::{PreprocessingError, Preprocessor};

/// Polynomial feature expansion
///
/// Expand input matrix with polynomial features
pub struct Polynomial {
    degree: usize,
    interaction_only: bool,
    input_dim: Option<usize>,
    output_dim: Option<usize>,
    combinations: Option<Vec<Vec<i32>>>,
    parallel_workers: usize,
}

impl Polynomial {
    pub fn new(degree: usize, interaction_only: bool, parallel_workers: usize) -> Polynomial {
        Polynomial {
            degree,
            interaction_only,
            input_dim: None,
            output_dim: None,
            combinations: None,
            parallel_workers,
        }
    }

    pub fn combinations(&self) -> &Option<Vec<Vec<i32>>> {
        &self.combinations
    }
}

impl Preprocessor<f32> for Polynomial {
    fn fit(
        &mut self,
        x: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<(), PreprocessingError> {
        let cols = x.shape()[1];
        self.input_dim = Some(cols);

        // Build combinations vector. Each entry contains the ids of the x array
        // that are multiplied to create the i-th column in the transformed array
        let mut combinations: Vec<Vec<i32>> = Vec::new();
        for k in 1..=self.degree {
            let it = {
                if self.interaction_only {
                    (0..cols as i32).combinations(k).collect::<Vec<Vec<i32>>>()
                } else {
                    (0..cols as i32)
                        .combinations_with_replacement(k)
                        .collect::<Vec<Vec<i32>>>()
                }
            };
            combinations.extend(it);
        }
        self.output_dim = Some(combinations.len());
        self.combinations = Some(combinations);
        Ok(())
    }

    fn transform(
        &self,
        x: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<Array2<f32>, PreprocessingError> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(self.parallel_workers)
            .build()
            .unwrap();
        let rows = x.shape()[0];
        if let Some(cols) = self.output_dim {
            if let Some(combinations) = self.combinations.as_ref() {
                let poly_x: RwLock<Array2<f32>> = RwLock::new(Array2::default((rows, cols)));
                pool.install(|| {
                    combinations
                        .par_iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let mut current_col = x.slice(s![.., c[0]]).to_owned();
                            for el in c.iter().skip(1) {
                                current_col = current_col.mul(&x.slice(s![.., *el]));
                            }
                            poly_x.write().unwrap().column_mut(i).assign(&current_col);
                        })
                        .collect::<Vec<()>>()
                });
                // Move out the value from the RwLock
                // we get the exclusive access from the RwLock and then we replace the cotent with another array object
                // getting back the original one that we can return to the caller
                let mut write_lock = poly_x.write().unwrap();
                let moved_value = std::mem::replace(&mut *write_lock, Array2::<f32>::zeros((3, 3)));
                Ok(moved_value)
            } else {
                Err(PreprocessingError::TransformError)
            }
        } else {
            Err(PreprocessingError::TransformError)
        }
    }

    fn inverse_transform(
        &self,
        _x: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<Array2<f32>, PreprocessingError> {
        Err(PreprocessingError::TransformError)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    use crate::preprocessing::{polynomial::Polynomial, Preprocessor};

    #[test]
    fn polynomial() {
        let n = 2;
        let degree = 3;
        let r = 2;
        let x = Array2::from_shape_fn((r, n), |(i, j)| (1.0 + i as f32) * (1.0 + j as f32));
        let correct_poly = Array2::from(vec![
            //   x    y     xx, xy,  yy,   xxx   xxy xyy, yyy
            [1.0, 2.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 8.0],
            [2.0, 4.0, 4.0, 8.0, 16.0, 8.0, 16.0, 32.0, 64.0],
        ]);
        let mut transformer = Polynomial::new(degree, false, 12);
        let _ = transformer.fit(&x.view());
        let result = transformer.transform(&x.view()).unwrap();
        assert_abs_diff_eq!(result, correct_poly, epsilon = 1e-3);
    }
}
