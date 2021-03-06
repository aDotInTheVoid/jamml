// Copyright 2018 Nixon Enraght-Moony

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Provides functions to create matricies
//!
//! ```
//! # use jamml::initializers::column_mat;
//! let x = column_mat(&vec![1, 2, 3]);
//! let y = vec![vec![1],
//!              vec![2],
//!              vec![3]];
//!
//! assert_eq!(x, y);
//! ```

extern crate num_traits;
use num_traits::NumAssign;

extern crate rand;
// Dont remove `Rng` useage, it breaks stuff.
use rand::{thread_rng, Rng};

use crate::core::Mat;

/// Creates a matrix for a column vector from `&Vec v`
///
/// Essensialt we take each element in the input vector and put it into its own vector,
/// before stringing them together.
/// ```text
///              [[a]
/// [a, b, c] ->  [b]
///               [c]]
/// ```
pub fn column_mat<T: NumAssign + Copy>(v: &Vec<T>) -> Mat<T> {
    let mut m: Mat<T> = Vec::new();

    for i in v {
        m.push(vec![*i]);
    }
    return m;
}

/// Creates a `m` by `n` matrix of type `T`, where each element is `T::zero()`
///
/// ```rust
/// # use jamml::initializers::zero_mat;
/// let x = zero_mat::<i32>(3, 2);
/// let y = vec![vec![0, 0],
///              vec![0, 0],
///              vec![0, 0]];
/// assert_eq!(x, y);
/// ```
pub fn zero_mat<T: NumAssign + Copy>(m: usize, n: usize) -> Mat<T> {
    n_mat(m, n, T::zero())
}

/// Creates a `m` by `n` matrix of type `T`, where each element is `T::one()`
///
/// ```rust
/// # use jamml::initializers::one_mat;
/// let x = one_mat::<i32>(3, 2);
/// let y = vec![vec![1, 1],
///              vec![1, 1],
///              vec![1, 1]];
/// assert_eq!(x, y)
/// ```
pub fn one_mat<T: NumAssign + Copy>(m: usize, n: usize) -> Mat<T> {
    n_mat(m, n, T::one())
}

/// Creates a `m` by `n` matrix of type `T`, where each element is `x`
///
/// ```rust
/// # use jamml::initializers::n_mat;
/// let x = n_mat(3, 2, 7);
/// let y = vec![vec![7, 7],
///              vec![7, 7],
///              vec![7, 7]];
/// assert_eq!(x, y);
/// ```
pub fn n_mat<T: NumAssign + Copy>(m: usize, n: usize, x: T) -> Mat<T> {
    //TODO: Return result or use non zero type
    assert!(m != 0 && n != 0);
    // I think `vec!` allocs enough capacity
    //TODO: check if vec! is optimised enough
    vec![vec![x; n]; m]
}

/// Creates a `m` by `m` identity matrix of type `T`
///
/// ```rust
/// # use jamml::initializers::identity_mat;
/// let x = identity_mat::<i32>(4);
/// let y = vec![vec![1, 0, 0, 0],
///              vec![0, 1, 0, 0],
///              vec![0, 0, 1, 0],
///              vec![0, 0, 0, 1]];
/// assert_eq!(x, y);
/// ```
pub fn identity_mat<T: NumAssign + Copy>(m: usize) -> Mat<T> {
    let mut r = zero_mat(m, m);
    for i in 0..m {
        r[i][i] = T::one();
    }
    return r;
}

/// Creates a `m` by `n` matrix of by using function `f` to determine each element.
///
/// `f` shound be a function that takes no arguments and return `T`
///
/// ```rust
/// # use jamml::initializers::fn_mat;
/// let x = fn_mat(4, 2, ||{2*4});
/// let y = vec![vec![8, 8],
///              vec![8, 8],
///              vec![8, 8],
///              vec![8, 8]];
/// assert_eq!(x, y);
/// ```
pub fn fn_mat<T: NumAssign + Copy, F: Fn() -> T>(m: usize, n: usize, f: F) -> Mat<T> {
    let mut r = zero_mat(m, n);
    for i in 0..m {
        for j in 0..n {
            r[i][j] = f();
        }
    }
    return r;
}

/// Creates a `m` by `n` matrix of random values between `min` and `max`
///
/// Panics if `min >= max`
pub fn ranged_rand_mat<T>(m: usize, n: usize, min: T, max: T) -> Mat<T>
where
    T: NumAssign + Copy + rand::distributions::uniform::SampleUniform,
{
    fn_mat(m, n, || rand::thread_rng().gen_range(min, max))
}

/// Creates a `m` by `n` matrix of random values between `val` and `-val`
pub fn ranged_rand_around_mat<T>(m: usize, n: usize, val: T) -> Mat<T>
where
    T: NumAssign + Copy + rand::distributions::uniform::SampleUniform + num_traits::sign::Signed,
{
    let val = val.abs();
    ranged_rand_mat(m, n, -val, val)
}

/// Creates a `m` by `n` matrix of random values between `1` and `-1`
pub fn one_to_minus_one_mat<T>(m: usize, n: usize) -> Mat<T>
where
    T: NumAssign + Copy + rand::distributions::uniform::SampleUniform + num_traits::sign::Signed,
{
    ranged_rand_mat(m, n, -T::one(), T::one())
}

#[cfg(test)]
mod tests {
    use super::*;

    mod column_vecs {
        use super::*;
        #[test]
        fn create_column_vec() {
            let v: Mat<i32> = column_mat(&vec![1, 2, 3]);
            assert_eq!(v, vec![vec![1], vec![2], vec![3]])
        }
        #[test]
        fn column_vec_4() {
            let mut rng = thread_rng();
            for _ in 1..10 {
                let a = rng.gen_range(0, 10);
                let b = rng.gen_range(0, 10);
                let c = rng.gen_range(0, 10);
                let d = rng.gen_range(0, 10);

                let v = column_mat(&vec![a, b, c, d]);
                let w = vec![vec![a], vec![b], vec![c], vec![d]];
                assert_eq!(v, w);
            }
        }
        #[test]
        fn column_vec_3() {
            let mut rng = thread_rng();
            for _ in 1..10 {
                let a = rng.gen_range(0, 10);
                let b = rng.gen_range(0, 10);
                let c = rng.gen_range(0, 10);

                let v = column_mat(&vec![a, b, c]);
                let w = vec![vec![a], vec![b], vec![c]];
                assert_eq!(v, w);
            }
        }
        #[test]
        fn column_vec_2() {
            let mut rng = thread_rng();
            for _ in 1..10 {
                let a = rng.gen_range(0, 10);
                let b = rng.gen_range(0, 10);

                let v = column_mat(&vec![a, b]);
                let w = vec![vec![a], vec![b]];
                assert_eq!(v, w);
            }
        }
        #[test]
        fn column_vec_1() {
            let mut rng = thread_rng();
            for _ in 1..10 {
                let a = rng.gen_range(0, 10);

                let v = column_mat(&vec![a]);
                let w = vec![vec![a]];
                assert_eq!(v, w);
            }
        }
    }
    mod dimm_error_panics {
        use super::*;
        #[test]
        #[should_panic]
        fn zero_mat_panics_left_zero() {
            zero_mat::<i32>(0, 1);
        }
        #[test]
        #[should_panic]
        fn zero_mat_panics_right_zero() {
            zero_mat::<i32>(1, 0);
        }
        #[test]
        #[should_panic]
        fn zero_mat_panics_two_zeros() {
            zero_mat::<i32>(0, 0);
        }
        #[test]
        #[should_panic]
        fn one_mat_panics_left_zero() {
            one_mat::<i32>(0, 1);
        }
        #[test]
        #[should_panic]
        fn one_mat_panics_right_zero() {
            one_mat::<i32>(1, 0);
        }
        #[test]
        #[should_panic]
        fn one_mat_panics_two_zeros() {
            one_mat::<i32>(0, 0);
        }

        #[test]
        #[should_panic]
        fn n_mat_panics_left_zero() {
            n_mat::<i32>(0, 1, 1);
        }
        #[test]
        #[should_panic]
        fn n_mat_panics_right_zero() {
            n_mat::<i32>(1, 0, 1);
        }
        #[test]
        #[should_panic]
        fn n_mat_panics_two_zeros() {
            n_mat::<i32>(0, 0, 1);
        }
    }
    // TODO: Fix misleading internal variable names
    mod fixed_elem_mat_init {
        use super::*;
        mod one_by_n {
            use super::*;
            #[test]
            fn one_by_one() {
                let one_by_one_by_zero = vec![vec![0]];
                let one_by_one_by_one = vec![vec![1]];
                let one_by_one_by_73 = vec![vec![73]];

                assert_eq!(one_by_one_by_zero, zero_mat(1, 1));
                assert_eq!(one_by_one_by_one, one_mat(1, 1));
                assert_eq!(one_by_one_by_73, n_mat(1, 1, 73));
            }
            #[test]
            fn one_by_two() {
                let one_by_one_by_zero = vec![vec![0; 2]];
                let one_by_one_by_one = vec![vec![1; 2]];
                let one_by_one_by_73 = vec![vec![73; 2]];

                assert_eq!(one_by_one_by_zero, zero_mat(1, 2));
                assert_eq!(one_by_one_by_one, one_mat(1, 2));
                assert_eq!(one_by_one_by_73, n_mat(1, 2, 73));
            }
            #[test]
            fn one_by_three() {
                let one_by_one_by_zero = vec![vec![0; 3]];
                let one_by_one_by_one = vec![vec![1; 3]];
                let one_by_one_by_73 = vec![vec![73; 3]];

                assert_eq!(one_by_one_by_zero, zero_mat(1, 3));
                assert_eq!(one_by_one_by_one, one_mat(1, 3));
                assert_eq!(one_by_one_by_73, n_mat(1, 3, 73));
            }
        }
        mod two_by_n {
            use super::*;
            #[test]
            fn two_by_one() {
                let one_by_one_by_zero = vec![vec![0]; 2];
                let one_by_one_by_one = vec![vec![1]; 2];
                let one_by_one_by_73 = vec![vec![73]; 2];

                assert_eq!(one_by_one_by_zero, zero_mat(2, 1));
                assert_eq!(one_by_one_by_one, one_mat(2, 1));
                assert_eq!(one_by_one_by_73, n_mat(2, 1, 73));
            }
            #[test]
            fn two_by_two() {
                let one_by_one_by_zero = vec![vec![0; 2]; 2];
                let one_by_one_by_one = vec![vec![1; 2]; 2];
                let one_by_one_by_73 = vec![vec![73; 2]; 2];

                assert_eq!(one_by_one_by_zero, zero_mat(2, 2));
                assert_eq!(one_by_one_by_one, one_mat(2, 2));
                assert_eq!(one_by_one_by_73, n_mat(2, 2, 73));
            }
            #[test]
            fn two_by_three() {
                let one_by_one_by_zero = vec![vec![0; 3]; 2];
                let one_by_one_by_one = vec![vec![1; 3]; 2];
                let one_by_one_by_73 = vec![vec![73; 3]; 2];

                assert_eq!(one_by_one_by_zero, zero_mat(2, 3));
                assert_eq!(one_by_one_by_one, one_mat(2, 3));
                assert_eq!(one_by_one_by_73, n_mat(2, 3, 73));
            }
        }
        mod three_by_n {
            use super::*;
            #[test]
            fn three_by_one() {
                let one_by_one_by_zero = vec![vec![0]; 3];
                let one_by_one_by_one = vec![vec![1]; 3];
                let one_by_one_by_73 = vec![vec![73]; 3];

                assert_eq!(one_by_one_by_zero, zero_mat(3, 1));
                assert_eq!(one_by_one_by_one, one_mat(3, 1));
                assert_eq!(one_by_one_by_73, n_mat(3, 1, 73));
            }
            #[test]
            fn three_by_two() {
                let one_by_one_by_zero = vec![vec![0; 2]; 3];
                let one_by_one_by_one = vec![vec![1; 2]; 3];
                let one_by_one_by_73 = vec![vec![73; 2]; 3];

                assert_eq!(one_by_one_by_zero, zero_mat(3, 2));
                assert_eq!(one_by_one_by_one, one_mat(3, 2));
                assert_eq!(one_by_one_by_73, n_mat(3, 2, 73));
            }
            #[test]
            fn three_by_three() {
                let one_by_one_by_zero = vec![vec![0; 3]; 3];
                let one_by_one_by_one = vec![vec![1; 3]; 3];
                let one_by_one_by_73 = vec![vec![73; 3]; 3];

                assert_eq!(one_by_one_by_zero, zero_mat(3, 3));
                assert_eq!(one_by_one_by_one, one_mat(3, 3));
                assert_eq!(one_by_one_by_73, n_mat(3, 3, 73));
            }
        }
    }

    mod vario_elem_mat_init {
        use super::*;
        #[test]
        fn identity_one_to_four() {
            let i1 = vec![vec![1]];
            let i2 = vec![vec![1, 0], vec![0, 1]];
            let i3 = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
            let i4 = vec![
                vec![1, 0, 0, 0],
                vec![0, 1, 0, 0],
                vec![0, 0, 1, 0],
                vec![0, 0, 0, 1],
            ];

            assert_eq!(i1, identity_mat(1));
            assert_eq!(i2, identity_mat(2));
            assert_eq!(i3, identity_mat(3));
            assert_eq!(i4, identity_mat(4));
        }

        #[test]
        fn fn_mat_has_right_dims() {
            // TODO: Find a better way to test
            use crate::core::dims;
            let x = fn_mat(2, 2, || thread_rng().gen_range(1, 10));
            assert_eq!(dims(&x), (2, 2));
            let y = fn_mat(8, 2, || thread_rng().gen::<f32>());
            assert_eq!(dims(&y), (8, 2));
        }

        #[test]
        fn fn_mat_const_fn() {
            let x = fn_mat(4, 6, || 42);
            let y = vec![
                vec![42, 42, 42, 42, 42, 42],
                vec![42, 42, 42, 42, 42, 42],
                vec![42, 42, 42, 42, 42, 42],
                vec![42, 42, 42, 42, 42, 42],
            ];
            assert_eq!(x, y);
        }

        #[test]
        fn ranged_rand_mat_is_ranged() {
            let x = ranged_rand_mat(10, 10, 0, 10);
            for i in 0..10 {
                for j in 0..10 {
                    assert!(0 <= x[i][j]);
                    assert!(x[i][j] <= 10);
                }
            }
        }

        #[test]
        #[should_panic]
        fn ranged_rand_mat_panics_on_wrong_order() {
            ranged_rand_mat(10, 10, 10, 0);
        }

        #[test]
        fn ranged_rand_around_mat_is_ranged_around() {
            let x = ranged_rand_around_mat(10, 10, 10);
            for i in 0..10 {
                for j in 0..10 {
                    assert!(-10 <= x[i][j]);
                    assert!(x[i][j] <= 10);
                }
            }
            // Test negitive case
            let x = ranged_rand_around_mat(10, 10, -10);
            for i in 0..10 {
                for j in 0..10 {
                    assert!(-10 <= x[i][j]);
                    assert!(x[i][j] <= 10);
                }
            }
        }

        #[test]
        fn one_to_minus_one_mat_is_one_to_minus_one() {
            let x = one_to_minus_one_mat::<f32>(10, 10);
            for i in 0..10 {
                for j in 0..10 {
                    assert!(-10.0 <= x[i][j]);
                    assert!(x[i][j] <= 10.0);
                }
            }
        }
    }
}
