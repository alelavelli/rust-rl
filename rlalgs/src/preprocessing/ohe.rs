use ndarray::{Array1, Array2, ArrayBase, Axis, Dim, ViewRepr};
use ndarray_stats::QuantileExt;

use super::{PreprocessingError, Preprocessor};

/// One Hot Encoder
///
/// Transform the categorical value into numerical matrix of
/// 1 and zeros.
pub struct OneHotEncoder {
    num_values: Option<usize>,
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new(None)
    }
}

impl OneHotEncoder {
    /// Create new struct instance
    pub fn new(num_values: Option<usize>) -> OneHotEncoder {
        OneHotEncoder { num_values }
    }

    pub fn transform_elem(&self, x: &i32) -> Result<Array1<f32>, PreprocessingError> {
        if let Some(num_values) = self.num_values {
            if *x as usize >= num_values {
                return Err(PreprocessingError::TransformError);
            }
            let mut array = Array1::zeros(num_values);
            array[*x as usize] = 1.0;
            Ok(array)
        } else {
            Err(PreprocessingError::TransformError)
        }
    }
}

impl Preprocessor<i32> for OneHotEncoder {
    fn fit(
        &mut self,
        x: &ArrayBase<ViewRepr<&i32>, Dim<[usize; 2]>>,
    ) -> Result<(), PreprocessingError> {
        // num_values is equal to the maximum number in x plus 1 because of the presence of 0
        self.num_values = Some(*x.max().unwrap() as usize + 1);
        Ok(())
    }

    fn transform(
        &self,
        x: &ArrayBase<ViewRepr<&i32>, Dim<[usize; 2]>>,
    ) -> Result<Array2<f32>, PreprocessingError> {
        if let Some(num_values) = self.num_values {
            if *x.max().unwrap() as usize > num_values {
                return Err(PreprocessingError::TransformError);
            }
            let mut matrix = Array2::zeros((x.shape()[0], num_values));
            // naive implementation
            for (i, r) in x.iter().enumerate() {
                matrix[[i, *r as usize]] = 1.0;
            }
            Ok(matrix)
        } else {
            Err(PreprocessingError::TransformError)
        }
    }

    fn inverse_transform(
        &self,
        x: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>,
    ) -> Result<Array2<i32>, PreprocessingError> {
        Ok(x.map_axis(Axis(0), |row| row.argmax().unwrap() as i32)
            .insert_axis(Axis(1)))
    }
}

#[cfg(test)]
mod tests {
    use crate::preprocessing::Preprocessor;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    use super::OneHotEncoder;

    #[test]
    fn ohe() {
        let num_values = 3;
        let mut encoder = OneHotEncoder::new(Some(num_values));
        let mut encoder_default = OneHotEncoder::default();

        let x = Array2::from_shape_vec((3, 1), vec![0, 1, 2]).unwrap();

        _ = encoder.fit(&x.view());
        _ = encoder_default.fit(&x.view());

        for e in [encoder, encoder_default].iter() {
            let result = e.transform(&x.view());
            assert!(result.is_ok());
            let unwrapped_result = result.unwrap();
            let exact_result =
                Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                    .unwrap();
            assert_abs_diff_eq!(unwrapped_result, exact_result, epsilon = 1e-3);

            let elem = 1;
            let elem_result = e.transform_elem(&elem);
            let exact_elem_result = Array1::from_vec(vec![0.0, 1.0, 0.0]);
            assert!(elem_result.is_ok());
            assert_eq!(elem_result.unwrap(), exact_elem_result);

            let x_wrong = Array2::from_shape_vec((3, 1), vec![4, 5, 6]).unwrap();
            let wrong_result = e.transform(&x_wrong.view());
            assert!(wrong_result.is_err());
        }
    }
}
