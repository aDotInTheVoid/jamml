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

extern crate num_traits;
use num_traits::NumAssign;

use crate::core;
use crate::core::Mat;

/// Calculates the dot product of two Vectors of Numbers.
pub fn dot_product<T: NumAssign + Copy>(a: &Vec<T>, b: &Vec<T>) -> T {
    let mut acc: T = T::zero();
    assert_eq!(a.len(), b.len());
    for (av, bv) in a.iter().zip(b.iter()) {
        acc += (*av) * (*bv);
    }
    return acc;
}

/// Calculates the transpose of a matrix
pub fn transpose<T: NumAssign + Copy>(a: &Mat<T>) -> Mat<T> {
    let (m, n): (usize, usize) = core::dims(a);
    let mut r: Mat<T> = Vec::new();

    // Give `r` some capacity
    for _ in 0..n {
        r.push(vec![T::zero(); m])
    }

    for i in 0..m {
        for j in 0..n {
            r[j][i] = a[i][j];
        }
    }

    return r;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};
    #[test]
    fn dot_product_2_elem() {
        assert_eq!(dot_product(&vec![2, 5], &vec![3, 1]), 11);
        assert_eq!(dot_product(&vec![4, 3], &vec![3, 5]), 27);
    }

    #[test]
    fn dot_product_3_elem() {
        assert_eq!(dot_product(&vec![1, 3, -5], &vec![4, -2, -1]), 3);
        assert_eq!(dot_product(&vec![3, 1, 8], &vec![4, 2, 3]), 38);
        assert_eq!(dot_product(&vec![2, 5, -2], &vec![1, 8, -3]), 48);
    }

    #[test]
    fn transpose_2x2() {
        let mut rng = thread_rng();
        for _ in 1..10 {
            let a = rng.gen_range(0, 10);
            let b = rng.gen_range(0, 10);
            let c = rng.gen_range(0, 10);
            let d = rng.gen_range(0, 10);

            // [[a b]
            //  [c d]]
            let x = vec![vec![a, b], vec![c, d]];

            // [[a c]
            //  [b d]
            let y = vec![vec![a, c], vec![b, d]];

            assert_eq!(x, transpose(&y));
        }
    }
    #[test]
    fn transpose_3x3() {
        let mut rng = thread_rng();
        for _ in 1..10 {
            let a = rng.gen_range(0, 10);
            let b = rng.gen_range(0, 10);
            let c = rng.gen_range(0, 10);
            let d = rng.gen_range(0, 10);
            let e = rng.gen_range(0, 10);
            let f = rng.gen_range(0, 10);
            let g = rng.gen_range(0, 10);
            let h = rng.gen_range(0, 10);
            let i = rng.gen_range(0, 10);

            // [[a b c]
            //  [d e f]
            //  [g h i]]
            let x = vec![vec![a, b, c], vec![d, e, f], vec![g, h, i]];

            // [[a d g]
            //  [b e h]
            //  [c f i]
            let y = vec![vec![a, d, g], vec![b, e, h], vec![c, f, i]];

            assert_eq!(x, transpose(&y));
        }
    }
    #[test]
    fn transpose_2x3() {
        let mut rng = thread_rng();
        for _ in 1..10 {
            let a = rng.gen_range(0, 10);
            let b = rng.gen_range(0, 10);
            let c = rng.gen_range(0, 10);
            let d = rng.gen_range(0, 10);
            let e = rng.gen_range(0, 10);
            let f = rng.gen_range(0, 10);

            // [[a b c]
            //  [d e f]
            let x = vec![vec![a, b, c], vec![d, e, f]];

            // [[a d]
            //  [b e]
            //  [c f]]
            let y = vec![vec![a, d], vec![b, e], vec![c, f]];

            assert_eq!(x, transpose(&y));
        }
    }
    #[test]
    fn transpose_3x2() {
        let mut rng = thread_rng();
        for _ in 1..10 {
            let a = rng.gen_range(0, 10);
            let b = rng.gen_range(0, 10);
            let d = rng.gen_range(0, 10);
            let e = rng.gen_range(0, 10);
            let g = rng.gen_range(0, 10);
            let h = rng.gen_range(0, 10);

            // [[a b]
            //  [d e]
            //  [g h]]
            let x = vec![vec![a, b], vec![d, e], vec![g, h]];

            // [[a d g]
            //  [b e h]]
            let y = vec![vec![a, d, g], vec![b, e, h]];

            assert_eq!(x, transpose(&y));
        }
    }
}
