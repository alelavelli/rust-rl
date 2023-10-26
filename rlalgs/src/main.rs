use ndarray;
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use rlalgs::{
    regressor::{linear::LinearRegression, Regressor},
    value_function::StateActionValueFunction,
};

fn main() {
    let n_samples = 100;
    let x = ndarray::Array::random((n_samples, 2), Uniform::new(-1., 1.));
    let w = ndarray::Array::from_shape_vec((2, 1), vec![0.5, 0.1]).unwrap();
    let b = 0.0;
    let y = &x.dot(&w) + b;
    println!("shape of y {:?}", y.shape());
    let mut linreg = LinearRegression::new(2, 0.05);
    println!("initial weights are {:?}", linreg.weights);
    linreg.fit(&x, &y);
    println!("learnt weights are {:?}", linreg.weights);

    let mut linreg = LinearRegression::new(2, 0.05);
    println!("initial weights are {:?}", linreg.weights);
    for _ in 0..5 {
        for i in 0..(x.shape()[0]) {
            let sample_state = vec![x.column(0)[i]];
            let sample_action = vec![x.column(1)[i]];
            let sample_target = y.column(0)[i];
            linreg
                .update(&sample_state, &sample_action, sample_target)
                .unwrap();
        }
    }
    println!("learnt weights are {:?}", linreg.weights);
}
