use itertools::Itertools;
use ndarray::{s, Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};
use ndarray_stats::QuantileExt;
use rayon::{prelude::*, ThreadPoolBuilder};

use std::{fmt::Debug, ops::Mul, sync::RwLock};

#[derive(thiserror::Error, Debug)]
pub enum PreprocessingError {
    #[error("Failed to fit data")]
    FitError,

    #[error("Failed to transform data")]
    TransformError,
}

/// One Hot Encoder
///
/// Transform the categorical value into numerical matrix of
/// 1 and zeros.
pub struct OneHotEncoder {
    num_values: usize,
}

impl OneHotEncoder {
    /// Create new struct instance
    pub fn new(num_values: usize) -> OneHotEncoder {
        OneHotEncoder { num_values }
    }

    fn empty_array1(&self) -> Array1<f32> {
        Array1::zeros(self.num_values)
    }

    fn empty_array2(&self, rows: usize) -> Array2<f32> {
        Array2::zeros((rows, self.num_values))
    }

    pub fn transform(
        &self,
        x: &ArrayBase<ViewRepr<&i32>, Dim<[usize; 1]>>,
    ) -> Result<Array2<f32>, PreprocessingError> {
        if *x.max().unwrap() as usize > self.num_values {
            return Err(PreprocessingError::TransformError);
        }
        let mut matrix = self.empty_array2(self.num_values);
        // naive implementation
        for (i, r) in x.iter().enumerate() {
            matrix[[i, *r as usize]] = 1.0;
        }
        Ok(matrix)
    }

    pub fn transform_elem(&self, x: &i32) -> Result<Array1<f32>, PreprocessingError> {
        if *x as usize >= self.num_values {
            return Err(PreprocessingError::TransformError);
        }
        let mut array = self.empty_array1();
        array[*x as usize] = 1.0;
        Ok(array)
    }
}

/// Scale data with z-score
///
/// $$ \tilde{x} = \frac{x - \mu}{\sigma}$$
pub struct ZScore {
    means: Option<Array1<f32>>,
    stds: Option<Array1<f32>>,
}

impl Default for ZScore {
    fn default() -> Self {
        Self::new()
    }
}

impl ZScore {
    pub fn new() -> ZScore {
        ZScore {
            means: None,
            stds: None,
        }
    }

    // Learns processing from data
    pub fn fit(&mut self, x: &Array2<f32>) -> Result<&mut Self, PreprocessingError> {
        self.means = Some(x.mean_axis(Axis(0)).ok_or(PreprocessingError::FitError)?);
        self.stds = Some(x.std_axis(Axis(0), 1.0));
        Ok(self)
    }

    // Transform data according to parameters
    pub fn transform(&self, x: &Array2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        if let (Some(mu), Some(sigma)) = (self.means.as_ref(), self.stds.as_ref()) {
            Ok((x - mu) / sigma)
        } else {
            Err(PreprocessingError::TransformError)
        }
    }
}

/// Normalize data in the interval [a, b]
///
/// $$ \tilde{x} = a + \frac{(x - x_{min}) \times (a - b) }{x_{max} - x_{min}} $$
pub struct RangeNorm {
    a: f32,
    b: f32,
    x_min: Option<Array1<f32>>,
    x_max: Option<Array1<f32>>,
}

impl Default for RangeNorm {
    fn default() -> Self {
        Self::new(Some(0.0), Some(1.0))
    }
}

impl RangeNorm {
    pub fn new(a: Option<f32>, b: Option<f32>) -> RangeNorm {
        RangeNorm {
            a: a.unwrap_or(0.0),
            b: b.unwrap_or(1.0),
            x_min: None,
            x_max: None,
        }
    }

    pub fn fit(&mut self, x: &Array2<f32>) -> Result<&mut Self, PreprocessingError> {
        self.x_min = Some(x.map_axis(Axis(0), |view| {
            *view
                .into_iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }));
        self.x_max = Some(x.map_axis(Axis(0), |view| {
            *view
                .into_iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }));
        Ok(self)
    }

    pub fn transform(&self, x: &Array2<f32>) -> Result<Array2<f32>, PreprocessingError> {
        if let (Some(x_min), Some(x_max)) = (self.x_min.as_ref(), self.x_max.as_ref()) {
            Ok(self.a + (x - x_min) * (self.b - self.a) / (x_max - x_min))
        } else {
            Err(PreprocessingError::TransformError)
        }
    }
}

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

    pub fn fit(&mut self, x: Array2<f32>) -> Result<&mut Self, PreprocessingError> {
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
        Ok(self)
    }
    pub fn transform(&self, x: Array2<f32>) -> Result<Array2<f32>, PreprocessingError> {
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
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, Axis};
    use ndarray_rand::RandomExt;
    use rand_distr::{Normal, Uniform};

    use super::{OneHotEncoder, RangeNorm, ZScore};

    #[test]
    fn ohe() {
        let num_values = 3;
        let encoder = OneHotEncoder::new(num_values);

        let x = Array1::from_vec(vec![0, 1, 2]);
        let result = encoder.transform(&x.view());
        assert!(result.is_ok());
        let unwrapped_result = result.unwrap();
        let exact_result =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                .unwrap();
        assert_abs_diff_eq!(unwrapped_result, exact_result, epsilon = 1e-3);

        let elem = 1;
        let elem_result = encoder.transform_elem(&elem);
        let exact_elem_result = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        assert!(elem_result.is_ok());
        assert_eq!(elem_result.unwrap(), exact_elem_result);

        let x_wrong = Array1::from_vec(vec![4, 5, 6]);
        let wrong_result = encoder.transform(&x_wrong.view());
        assert!(wrong_result.is_err());
    }

    #[test]
    fn zscore() {
        let x = Array2::random((100, 2), Normal::new(1.0, 0.3).unwrap());
        let mut encoder = ZScore::new();
        _ = encoder.fit(&x);
        let x_norm = encoder.transform(&x).unwrap();
        assert_abs_diff_eq!(encoder.means.unwrap(), Array1::ones(2), epsilon = 1e-1);
        assert_abs_diff_eq!(
            encoder.stds.unwrap(),
            Array1::from_elem(2, 0.3),
            epsilon = 1e-1
        );
        assert_abs_diff_eq!(
            x_norm.mean_axis(Axis(0)).unwrap(),
            Array1::zeros(2),
            epsilon = 1e-1
        );
        assert_abs_diff_eq!(
            x_norm.std_axis(Axis(0), 1.0),
            Array1::ones(2),
            epsilon = 1e-1
        );
    }

    #[test]
    fn range_norm() {
        let x = Array2::random((100, 2), Uniform::new(5.0, 10.0));
        let mut encoder = RangeNorm::default();
        _ = encoder.fit(&x);
        println!("{:?} {:?}", encoder.x_min, encoder.x_max);
        let x_norm = encoder.transform(&x).unwrap();
        assert_abs_diff_eq!(
            x_norm.map_axis(Axis(0), |view| *view
                .into_iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()),
            Array1::zeros(2),
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            x_norm.map_axis(Axis(0), |view| *view
                .into_iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()),
            Array1::ones(2),
            epsilon = 1e-3
        );
    }
}
