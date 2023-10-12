use ndarray::{array, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::Rng;
use rand_distr::Uniform;

fn main() {
    let states = vec![[1, 2], [3, 4]];
    let actions = vec![[1,], [2]];
    let mut arr = Array2::zeros((2, 3));
    for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
        let sa = [&states[i][..], &actions[i][..]].concat();
        for (j, col) in row.iter_mut().enumerate() {
            *col = sa[j];
        }
    }
    println!("{:?}", arr);
}
