use ndarray::{Array, s};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray::prelude::*;

use rlenv;

fn main() {
    rlenv::hello_free();

    let mut a = Array::random((5, 2), Uniform::new(-1., 1.));

    println!("{}", a);
    
}
