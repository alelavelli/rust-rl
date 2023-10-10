use ndarray::array;

fn f(state: &Vec<f64>, action: &Vec<f64>) {
    let concat = [&state[..], &action[..]].concat();
    println!("{:?}", concat);
    let input = array!(concat);
    println!("{:?}", input.shape());
    println!("{:?}", input);
    let concat = [&state[..], &action[..]].concat();
    let input = ndarray::Array::from_shape_vec(
        (1, 4), concat
    ).unwrap();
    println!("{:?}", input.shape());
    println!("{:?}", input);
}

fn main() {
    let state = vec![1., 2.];
    let action = vec![3., 4.];
    f(&state, &action)
    
}
