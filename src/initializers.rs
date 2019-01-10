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
//! # use jamml::initializers::vec_to_column_mat;
//! let x = vec_to_column_mat(&vec![1, 2, 3]);
//! let y = vec![vec![1],
//!              vec![2],
//!              vec![3]];
//!
//! assert_eq!(x, y);
//! ```

extern crate num_traits;
use num_traits::NumAssign;

use crate::core::Mat;

/// Converts a vector of numbers to a matrix of a column vector.
///
/// Essensialt we take each element in the input vector and put it into its own vector,
/// before stringing them together.
/// ```text
///              [[a]
/// [a, b, c] ->  [b]
///               [c]]
/// ```
pub fn vec_to_column_mat<T: NumAssign + Copy>(v: &Vec<T>) -> Mat<T> {
    let mut m: Mat<T> = Vec::new();

    for i in v {
        m.push(vec![*i]);
    }
    return m;
}

/// Creates a `m` by `n` matrix of type `T`, where each element is `T::zero()`
pub fn zero_mat<T: NumAssign + Copy>(m: usize, n: usize) -> Mat<T> {
    n_mat(m, n, T::zero())
}

/// Creates a `m` by `n` matrix of type `T`, where each element is `T::one()`
pub fn one_mat<T: NumAssign + Copy>(m: usize, n: usize) -> Mat<T> {
    n_mat(m, n, T::one())
}

/// Creates a `m` by `n` matrix of type `T`, where each element is `x`
pub fn n_mat<T: NumAssign + Copy>(m: usize, n: usize, x: T) -> Mat<T> {
    //TODO: Return result or use non zero type
    assert!(m != 0 && n != 0);
    //TODO: Initialise vector with enough capacity
    let mut r: Mat<T> = Vec::new();
    for _ in 0..m {
        r.push(vec![x; n]);
    }
    return r;
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    use rand::{thread_rng, Rng};

    mod column_vecs {
        use super::*;
        #[test]
        fn create_column_vec() {
            let v: Mat<i32> = vec_to_column_mat(&vec![1, 2, 3]);
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

                let v = vec_to_column_mat(&vec![a, b, c, d]);
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

                let v = vec_to_column_mat(&vec![a, b, c]);
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

                let v = vec_to_column_mat(&vec![a, b]);
                let w = vec![vec![a], vec![b]];
                assert_eq!(v, w);
            }
        }
        #[test]
        fn column_vec_1() {
            let mut rng = thread_rng();
            for _ in 1..10 {
                let a = rng.gen_range(0, 10);

                let v = vec_to_column_mat(&vec![a]);
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
}
