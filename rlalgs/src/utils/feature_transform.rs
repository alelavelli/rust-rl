//! Transform the categorical value into a one hot encoding
//! vector with 1.0 in the position of the input value
pub fn to_ohe(value: usize, dim: usize) -> Vec<f32> {
    let mut ohe_array = vec![0.0; dim];
    ohe_array[value] = 1.0;
    ohe_array
}
