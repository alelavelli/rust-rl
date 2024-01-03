/*use std::{
    collections::HashMap,
    ops::Mul,
    sync::{Arc, RwLock, RwLockReadGuard},
    time::Instant,
};

use itertools::Itertools;
use ndarray::{concatenate, s, stack, Array1, Array2, Axis};
use ndarray_linalg::AllocatedArrayMut;
use rayon::{prelude::*, ThreadPoolBuilder};

fn naif(degree: usize, x: Array2<f32>) {
    let r = x.shape()[0];
    let n = x.shape()[1];
    let mut combinations: Vec<Vec<i32>> = Vec::new();
    for k in 1..=degree {
        let it = (0..n as i32).combinations_with_replacement(k);
        combinations.extend(it.collect::<Vec<Vec<i32>>>());
    }
    //println!("combinations\n{:?}", combinations);
    let mut poly_x: Array2<f32> = Array2::default((r, combinations.len()));

    for (i, c) in combinations.iter().enumerate() {
        let mut current_col = x.slice(s![.., c[0]]).to_owned();
        for el in c.iter().skip(1) {
            current_col = current_col.mul(&x.slice(s![.., *el]));
        }
        poly_x.column_mut(i).assign(&current_col);
    }
}

fn cached(degree: usize, x: Array2<f32>) -> Array2<f32> {
    let r = x.shape()[0];
    let n = x.shape()[1];
    let start = Instant::now();
    let mut combinations: Vec<Vec<i32>> = Vec::new();
    for k in 1..=degree {
        let it = (0..n as i32).combinations_with_replacement(k);
        combinations.extend(it.collect::<Vec<Vec<i32>>>());
    }
    //println!("end combinations {:?}", start.elapsed());
    //println!("combinations\n{:?}", combinations);

    let mut cache: HashMap<Vec<i32>, Array1<f32>> = HashMap::new();

    for (_, c) in combinations.iter().enumerate() {
        let current_col = {
            let split = c.split_last().unwrap();
            let elems = split.1;

            let elems = Vec::from_iter(elems.iter().map(|x| *x));
            if cache.contains_key(&elems) {
                cache.get(&elems).unwrap().mul(&x.slice(s![.., *split.0]))
            } else {
                let mut res = x.slice(s![.., c[0]]).to_owned();
                for el in c.iter().skip(1) {
                    res = res.mul(&x.slice(s![.., *el]));
                }
                res
            }
        };
        cache.insert(c.to_owned(), current_col.clone());
    }
    //println!("end cache {:?}", start.elapsed());
    let mut poly_x: Array2<f32> = Array2::default((r, combinations.len()));
    for (i, c) in combinations.iter().enumerate() {
        let current_col = cache.get(c).unwrap();
        poly_x.column_mut(i).assign(current_col);
    }
    poly_x
}

fn cached_v2(degree: usize, x: Array2<f32>) -> Array2<f32> {
    let r = x.shape()[0];
    let n = x.shape()[1];
    let start = Instant::now();
    let mut combinations: Vec<Vec<i32>> = Vec::new();
    for k in 1..=degree {
        let it = (0..n as i32).combinations_with_replacement(k);
        combinations.extend(it.collect::<Vec<Vec<i32>>>());
    }

    // cache contains indices of the matrix
    let mut poly_x: Array2<f32> = Array2::default((r, combinations.len()));
    let mut cache: HashMap<Vec<i32>, usize> = HashMap::new();
    for (i, c) in combinations.iter().enumerate() {
        let current_col = {
            let split = c.split_last().unwrap();
            let elems = split.1;

            let elems = Vec::from_iter(elems.iter().map(|x| *x));
            if cache.contains_key(&elems) {
                let index = cache.get(&elems).unwrap();
                poly_x.column(*index).mul(&x.column(*split.0 as usize))
            } else {
                let mut res = x.slice(s![.., c[0]]).to_owned();
                for el in c.iter().skip(1) {
                    res = res.mul(&x.column(*el as usize));
                }
                res
            }
        };
        poly_x.column_mut(i).assign(&current_col);
        cache.insert(c.to_owned(), i);
    }
    poly_x
    //println!("result \n{:?}", poly_x);
}

fn cached_v3(degree: usize, x: Array2<f32>) -> Array2<f32> {
    // concatenate arrays
    let r = x.shape()[0];
    let n = x.shape()[1];
    let start = Instant::now();
    let mut combinations: Vec<Vec<i32>> = Vec::new();
    for k in 1..=degree {
        let it = (0..n as i32).combinations_with_replacement(k);
        combinations.extend(it.collect::<Vec<Vec<i32>>>());
    }
    //println!("end combinations {:?}", start.elapsed());
    //println!("combinations\n{:?}", combinations);

    let mut first = true;
    let mut poly_x: Array2<f32> = Array2::default((r, 1));
    let mut cache: HashMap<Vec<i32>, Array1<f32>> = HashMap::new();

    for (_, c) in combinations.iter().enumerate() {
        let current_col = {
            let split = c.split_last().unwrap();
            let elems = split.1;

            let elems = Vec::from_iter(elems.iter().map(|x| *x));
            if cache.contains_key(&elems) {
                cache.get(&elems).unwrap().mul(&x.slice(s![.., *split.0]))
            } else {
                let mut res = x.slice(s![.., c[0]]).to_owned();
                for el in c.iter().skip(1) {
                    res = res.mul(&x.slice(s![.., *el]));
                }
                res
            }
        };
        if first {
            poly_x = current_col.insert_axis(Axis(1));
            first = false;
        } else {
            poly_x = concatenate![Axis(1), poly_x, current_col.insert_axis(Axis(1))];
        }
        //cache.insert(c.to_owned(), current_col);
    }
    //println!("end cache {:?}", start.elapsed());

    //println!("end matrix {:?}", start.elapsed());
    poly_x
    //println!("result \n{:?}", poly_x);
}

fn cached_par_v0(degree: usize, x: Array2<f32>) -> Array2<f32> {
    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let r = x.shape()[0];
    let n = x.shape()[1];
    let start = Instant::now();
    let mut combinations: Vec<Vec<i32>> = Vec::new();
    for k in 1..=degree {
        let it = (0..n as i32).combinations_with_replacement(k);
        combinations.extend(it.collect::<Vec<Vec<i32>>>());
    }
    //println!("end combinations {:?}", start.elapsed());
    //println!("combinations\n{:?}", combinations);

    let mut cache: HashMap<Vec<i32>, Array1<f32>> = HashMap::new();

    for (_, c) in combinations.iter().enumerate() {
        let current_col = {
            let split = c.split_last().unwrap();
            let elems = split.1;

            let elems = Vec::from_iter(elems.iter().map(|x| *x));
            if cache.contains_key(&elems) {
                cache.get(&elems).unwrap().mul(&x.slice(s![.., *split.0]))
            } else {
                let mut res = x.slice(s![.., c[0]]).to_owned();
                for el in c.iter().skip(1) {
                    res = res.mul(&x.slice(s![.., *el]));
                }
                res
            }
        };
        cache.insert(c.to_owned(), current_col.clone());
    }
    println!("end cache {:?}", start.elapsed());
    let mut poly_x: Array2<f32> = Array2::default((r, combinations.len()));

    pool.install(|| {
        for (i, c) in combinations.iter().enumerate() {
            let current_col = cache.get(c).unwrap();
            poly_x.column_mut(i).assign(current_col);
        }
    });

    println!("end matrix {:?}", start.elapsed());
    //println!("result \n{:?}", poly_x);
    poly_x
}

fn naif_v2(degree: usize, x: Array2<f32>) {
    let r = x.shape()[0];
    let n = x.shape()[1];
    let mut combinations: Vec<Vec<i32>> = Vec::new();
    for k in 1..=degree {
        let it = (0..n as i32)
            .combinations_with_replacement(k)
            .collect::<Vec<Vec<i32>>>();
        combinations.extend(it);
    }

    let mut poly_x: Array2<f32> = Array2::default((r, combinations.len()));
    let mut cache: HashMap<Vec<i32>, usize> = HashMap::new();

    for (i, c) in combinations.iter().enumerate() {
        let current_col = {
            let split = c.split_last().unwrap();
            let elems = split.1;

            let elems = Vec::from_iter(elems.iter().copied());
            if cache.contains_key(&elems) {
                poly_x
                    .slice(s![.., *cache.get(&elems).unwrap()])
                    .mul(&x.slice(s![.., *split.0]))
            } else {
                let mut res = x.slice(s![.., c[0]]).to_owned();
                for el in c.iter().skip(1) {
                    res = res.mul(&x.slice(s![.., *el]));
                }
                res
            }
        };
        cache.insert(c.to_owned(), i);
        poly_x.column_mut(i).assign(&current_col);
    }
}

fn naif_par(degree: usize, x: Array2<f32>) -> Array2<f32> {
    let pool = ThreadPoolBuilder::new().num_threads(12).build().unwrap();

    let r = x.shape()[0];
    let n = x.shape()[1];
    let mut combinations: Vec<Vec<i32>> = Vec::new();
    for k in 1..=degree {
        let it = (0..n as i32).combinations_with_replacement(k);
        combinations.extend(it.collect::<Vec<Vec<i32>>>());
    }

    //let poly_x: Arc<RwLock<Array2<f32>>> = Arc::new(RwLock::new(Array2::default((r, combinations.len()))));
    let poly_x: RwLock<Array2<f32>> = RwLock::new(Array2::default((r, combinations.len())));
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
    std::mem::replace(&mut *write_lock, Array2::<f32>::zeros((3, 3)))
}

fn main() {
    let n = 10;
    let degree = 8;
    let r = 5000;
    let x = Array2::from_shape_fn((r, n), |(i, j)| (1.0 + i as f32) * (1.0 + j as f32));
    let x1 = x.clone();
    let x2 = x.clone();
    //println!("x is \n {:?}", x);
    let start = Instant::now();
    naif(degree, x.clone());
    let duration = start.elapsed();
    println!("Time elapsed in naif() is: {:?}", duration);

    let start = Instant::now();
    let res = cached(degree, x1);
    let duration = start.elapsed();
    println!("Time elapsed in cache() is: {:?}", duration);

    let start = Instant::now();
    let res = naif_par(degree, x2);
    let duration = start.elapsed();
    println!("Time elapsed in naif_par() is: {:?}", duration);
    //println!("{:?}", res);
}
*/

use ndarray::{concatenate, Array1, Axis};

trait MyTrait {
    fn do_something(&self);
}

struct BaseObj {
    a: i32,
}

impl BaseObj {
    fn new(a: i32) -> BaseObj {
        BaseObj { a }
    }
}

impl MyTrait for BaseObj {
    fn do_something(&self) {
        println!("{}", self.a);
    }
}

struct DecoratorObj<T> {
    inner: T,
}

impl<T> DecoratorObj<T> {
    fn new(inner: T) -> DecoratorObj<T> {
        DecoratorObj { inner }
    }
}

impl<T: MyTrait> MyTrait for DecoratorObj<T> {
    fn do_something(&self) {
        println!("Decorated");
        self.inner.do_something();
    }
}

fn main() {
    let obj = DecoratorObj::new(BaseObj::new(2));
    obj.do_something();

    let a: Array1<f32> = Array1::zeros(5);
    let b: Array1<f32> = Array1::zeros(0);
    println!("a: {:?},\nb: {:?}", a, b);
    let c = concatenate(Axis(0), &[a.view(), b.view()]).unwrap();
    println!("c: {:?}", c);
}
