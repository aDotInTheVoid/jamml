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

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    use rand::{thread_rng, Rng};
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
