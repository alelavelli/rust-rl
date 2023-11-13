use ndarray::{Array1, Array2, ArrayBase, Dim, ViewRepr};

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

    pub fn transform(&self, x: &ArrayBase<ViewRepr<&i32>, Dim<[usize; 1]>>) -> Array2<f32> {
        let mut matrix = self.empty_array2(x.len());
        // naive implementation
        for (i, r) in x.iter().enumerate() {
            matrix[[i, *r as usize]] = 1.0;
        }
        matrix
    }

    pub fn transform_elem(&self, x: &i32) -> Array1<f32> {
        let mut array = self.empty_array1();
        array[*x as usize] = 1.0;
        array
    }
}
