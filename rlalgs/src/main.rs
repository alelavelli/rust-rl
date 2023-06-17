use ndarray::{s, Array, Array2, ArrayView, AssignElem};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rlalgs::{policy::EGreedyTabularPolicy, TabularPolicy};
use rlenv;

fn main() {
    rlenv::hello_free();

    let mut a: Array2<f64> = Array::random((2, 5), Uniform::new(-1., 1.));
    println!("{}", a);
    let elem = Array::from(vec![500.0]);
    a[[0, 0]] = 500.0;
    //a.slice_mut(s![0, 0]).assign_elem(&ArrayView::from(&[500.0]));
    //a.slice_mut(s![0, 0]).assign(&elem);
    println!("{}", a);

    let mut rng = rand::thread_rng();
    let pi = EGreedyTabularPolicy::new(5, 7, 0.5);
    let v: Vec<i32> = (0..10).rev().collect();
    println!("{:?}", v);
    println!("{}", pi.q);
    for i in 0..10 {
        println!("{}", pi.step(1, &mut rng).unwrap());
    }
}
