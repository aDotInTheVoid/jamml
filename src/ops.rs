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

//! Defines Functions that opperate on matrices
//!
//! ```
//! # use jamml::ops::transpose;
//! # use jamml::core::Mat;
//! let m: Mat<i32> = vec![vec![1, 2, 3],
//!                        vec![4, 5, 6],
//!                        vec![7, 8, 9]];
//!
//! let t: Mat<i32> = vec![vec![1, 4, 7],
//!                        vec![2, 5, 8],
//!                        vec![3, 6, 9]];
//!
//! assert_eq!(m, transpose(&t).unwrap());
//! ```

extern crate num_traits;
use num_traits::NumAssign;

use crate::core;
use crate::core::{dims, Mat, MatrixError};
use crate::initializers;

/// Calculates the dot product of two Vectors of Numbers.
///
/// Returns `Ok(T) if `a.len() == b.len()`. Otherwise returns `Err(MatrixError::InvalidDims)`
pub fn dot_product<T>(a: &Vec<T>, b: &Vec<T>) -> Result<T, MatrixError>
where
    T: NumAssign + Copy,
{
    if a.len() == b.len() {
        let mut acc: T = T::zero();
        for (av, bv) in a.iter().zip(b.iter()) {
            acc += (*av) * (*bv);
        }
        return Ok(acc);
    } else {
        return Err(MatrixError::InvalidDims);
    }
}

/// Calculates the transpose of a matrix
///
/// Returns `Ok(Mat<T>)` if `jamml::core::isvalid(a)`. Otherwise returns `Err(MatrixError::InvalidDims)`
pub fn transpose<T>(a: &Mat<T>) -> Result<Mat<T>, MatrixError>
where
    T: NumAssign + Copy,
{
    let (m, n): (usize, usize) = core::dims(a)?;
    let mut r: Mat<T> = Vec::new();

    // Give `r` some capacity
    for _ in 0..n {
        r.push(vec![T::zero(); m]);
    }

    for i in 0..m {
        for j in 0..n {
            r[j][i] = a[i][j];
        }
    }

    return Ok(r);
}

/// Multiplys matrix `a` by scalar `k`. Mutates value `a` through borrowing.
///
/// For a function that doesn't mutate `a` see `jamml::ops::scalar_mul`
///
/// Returns `Ok(Mat<T>)` if `jamml::core::isvalid(a)`. Otherwise returns `Err(MatrixError::NotRectangle)`
///
/// ```
/// # use jamml::ops::scalar_mul_inline;
/// let mut a = vec![vec![1,  8, -3],
///                  vec![4, -2,  5]];
/// let c = 2;
///
/// scalar_mul_inline(&mut a, c);
///
/// let ac = vec![vec![2, 16, -6],
///               vec![8, -4, 10]];
///
/// assert_eq!(a, ac)
/// ```
pub fn scalar_mul_inline<T>(
    a: &mut Mat<T>,
    k: T,
) -> Result<(), MatrixError>
where
    T: NumAssign + Copy,
{
    let (m, n) = dims(a)?;
    for i in 0..m {
        for j in 0..n {
            a[i][j] *= k;
        }
    }
    Ok(())
}

/// Returns matrix `a` times scalar `k`. Does not mutate `a` but clones
///
/// Note that this function clones `a` at runtime, so may be expensive for large
/// matrices
///
/// Returns `Ok(())` if `jamml::core::isvalid(a)`. Otherwise returns `Err(MatrixError::NotRectangle)`
///
/// For a function that avoids cloning by mutating `a` see `jamml::ops::scalar_mul_inline`
///
/// ```
/// # use jamml::ops::scalar_mul;
/// let a = vec![vec![1,  8, -3],
///              vec![4, -2,  5]];
/// let c = 2;
///
/// let ac = vec![vec![2, 16, -6],
///               vec![8, -4, 10]];
///
/// assert_eq!(scalar_mul(&a, c).unwrap(), ac)
/// ```
pub fn scalar_mul<T>(a: &Mat<T>, k: T) -> Result<Mat<T>, MatrixError>
where
    T: NumAssign + Copy,
{
    let mut m = a.clone();
    scalar_mul_inline(&mut m, k)?;
    Ok(m)
}

fn column<T>(a: &Mat<T>, n: usize) -> Vec<T>
where
    T: NumAssign + Copy,
{
    let mut r: Vec<T> = Vec::new();
    for i in a.iter() {
        r.push(i[n]);
    }
    return r;
}

/// Adds matrix `b` to matrix `a`. Mutates `a` to store results.
///
/// For a function that doesn't mutate `a` see `jamml::ops::add`
///
/// Returns `Ok(())` if no errors occur.
///
/// ## Errors
/// - If `a` or `b` isn't rectangular, `Err(MatrixError::NotRectangle)`
///  will be returned.
/// - If `dims(a) != dims(b)`, `Err(MatrixError::InvalidDims)` will be returned.
/// ```
/// # use jamml::ops::add_inline;
/// let mut a = vec![vec![0, 1, 2],
///                  vec![9, 8, 7]];
///
/// let b = vec![vec![6, 5, 4],
///              vec![3, 4, 5]];
///
/// add_inline(&mut a, &b);
///
/// let a_plus_b = vec![vec![6,  6,  6],
///                    vec![12, 12, 12]];
///
/// assert_eq!(a, a_plus_b);
/// ```
pub fn add_inline<T>(a: &mut Mat<T>, b: &Mat<T>) -> Result<(), MatrixError>
where
    T: NumAssign + Copy,
{
    // `a` is a `n` by `m` matrix.
    let (m, n) = dims(a)?;
    // b is a `n` by `m` matrix.
    if (m, n) != dims(b)? {
        return Err(MatrixError::InvalidDims);
    }

    for i in 0..m {
        for j in 0..n {
            a[i][j] += b[i][j];
        }
    }

    Ok(())
}

/// Adds matrix `b` to matrix `a`.
///
/// This function doesn't mutate values, but clones `a` and adds `b` to the cloned `a`, before returning the cloned `a`.
///
/// For a function that avoids cloning by mutating `a` see `jamml::ops::addmul_inline`
///
/// Returns `Ok(Mat<T>)` if no errors occur.
///
/// ## Errors
/// - If `a` or `b` isn't rectangular, `Err(MatrixError::NotRectangle)`
///  will be returned.
/// - If `dims(a) != dims(b)`, `Err(MatrixError::InvalidDims)` will be returned.
/// ```
/// # use jamml::ops::add;
/// let mut a = vec![vec![0, 1, 2],
///                  vec![9, 8, 7]];
///
/// let b = vec![vec![6, 5, 4],
///              vec![3, 4, 5]];
///
/// let x = add(&mut a, &b).unwrap();
///
/// let a_plus_b = vec![vec![6,  6,  6],
///                    vec![12, 12, 12]];
///
/// assert_eq!(a_plus_b, x);
/// ```
pub fn add<T>(a: &Mat<T>, b: &Mat<T>) -> Result<Mat<T>, MatrixError>
where
    T: NumAssign + Copy,
{
    let mut m = a.clone();
    add_inline(&mut m, b)?;
    Ok(m)
}

/// Multiplys matrix `a` by matrix `b`
///
/// Returns `Ok(Mat<T>)` if the arguments match up.
/// Note that the returned matrix uses newly allocated memory.
///
/// ```
/// let x = vec![vec![1, 2, 1],
///              vec![0, 1, 0],
///              vec![2, 3, 4]];
///
/// let y = vec![vec![2, 5],
///              vec![6, 7],
///              vec![1, 8]];
///
/// ```
/// ## Errors
/// - If `a` or `b` isn't rectangular, `Err(MatrixError::NotRectangle)` will be returned.
///
/// - If `a` or `b` is zero length, `Err(MatrixError::InvalidDims)`.
///
/// - If `dims(a)?.1 != dims(b)?.0` (the matrixes multiplication is
///   undefined because the dimensions dont line up),
///  `Err(MatrixError::InvalidDims)` will be returned
pub fn matmul<T>(a: &Mat<T>, b: &Mat<T>) -> Result<Mat<T>, MatrixError>
where
    T: NumAssign + Copy,
{
    // `a` is a `n` by `m` matrix.
    let (n, am) = dims(a)?;
    // b is a `m` by `p` matrix.
    let (bm, p) = dims(b)?;
    // check if the matrices have the same dimensions and are populated.
    if (am != bm) || (am == 0) {
        return Err(MatrixError::InvalidDims);
    }
    // this will check `n` and `p` don't equal `0`
    let mut ab = initializers::zero_mat(n, p)?;
    // core logic
    for i in 0..n {
        for j in 0..p {
            ab[i][j] = dot_product(&a[i], &column(b, j))
                .expect("something is horible wrong");
        }
    }

    return Ok(ab);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    mod dot_product_real {
        use super::*;
        #[test]
        fn dot_product_2_elem() {
            assert_eq!(dot_product(&vec![2, 5], &vec![3, 1]).unwrap(), 11);
            assert_eq!(dot_product(&vec![4, 3], &vec![3, 5]).unwrap(), 27);
        }

        #[test]
        fn dot_product_3_elem() {
            assert_eq!(
                dot_product(&vec![1, 3, -5], &vec![4, -2, -1]).unwrap(),
                3
            );
            assert_eq!(
                dot_product(&vec![3, 1, 8], &vec![4, 2, 3]).unwrap(),
                38
            );
            assert_eq!(
                dot_product(&vec![2, 5, -2], &vec![1, 8, -3]).unwrap(),
                48
            );
        }
    }
    mod transpose_rand {
        use super::*;

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

                assert_eq!(x, transpose(&y).unwrap());
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

                assert_eq!(x, transpose(&y).unwrap());
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

                assert_eq!(x, transpose(&y).unwrap());
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

                assert_eq!(x, transpose(&y).unwrap());
            }
        }
    }
    mod matmul_real {
        use super::*;

        #[test]
        fn matmul_2x2() {
            let a = vec![vec![1, 2], vec![0, 1]];
            let b = vec![vec![2, 5], vec![6, 7]];
            let ac_found = matmul(&a, &b).unwrap();
            let ac_real = vec![vec![14, 19], vec![6, 7]];
            assert_eq!(ac_found, ac_real);
        }
    }
    mod matmul_rand {
        use super::*;
        #[test]
        fn mat_symb_4x3_mul_3x2() {
            let mut rng = thread_rng();
            for _ in 0..10 {
                let a11 = rng.gen_range(-10, 10);
                let a12 = rng.gen_range(-10, 10);
                let a13 = rng.gen_range(-10, 10);
                let a21 = rng.gen_range(-10, 10);
                let a22 = rng.gen_range(-10, 10);
                let a23 = rng.gen_range(-10, 10);
                let a31 = rng.gen_range(-10, 10);
                let a32 = rng.gen_range(-10, 10);
                let a33 = rng.gen_range(-10, 10);
                let a41 = rng.gen_range(-10, 10);
                let a42 = rng.gen_range(-10, 10);
                let a43 = rng.gen_range(-10, 10);

                let a = vec![
                    vec![a11, a12, a13],
                    vec![a21, a22, a23],
                    vec![a31, a32, a33],
                    vec![a41, a42, a43],
                ];

                let b11 = rng.gen_range(-10, 10);
                let b12 = rng.gen_range(-10, 10);
                let b21 = rng.gen_range(-10, 10);
                let b22 = rng.gen_range(-10, 10);
                let b31 = rng.gen_range(-10, 10);
                let b32 = rng.gen_range(-10, 10);

                let b = vec![
                    // Padding
                    vec![b11, b12],
                    vec![b21, b22],
                    vec![b31, b32],
                ];

                let ans = vec![
                    vec![
                        a11 * b11 + a12 * b21 + a13 * b31,
                        a11 * b12 + a12 * b22 + a13 * b32,
                    ],
                    vec![
                        a21 * b11 + a22 * b21 + a23 * b31,
                        a21 * b12 + a22 * b22 + a23 * b32,
                    ],
                    vec![
                        a31 * b11 + a32 * b21 + a33 * b31,
                        a31 * b12 + a32 * b22 + a33 * b32,
                    ],
                    vec![
                        a41 * b11 + a42 * b21 + a43 * b31,
                        a41 * b12 + a42 * b22 + a43 * b32,
                    ],
                ];

                let ac = matmul(&a, &b).unwrap();

                assert_eq!(ans, ac);
            }
        }
    }
    mod add_rng {
        use super::*;
        #[test]
        fn rand_by_rand() {
            let mut rng = thread_rng();
            for _ in 0..20 {
                let m = rng.gen_range(1, 10);
                let n = rng.gen_range(1, 10);
                let a =
                    crate::initializers::ranged_rand_around_mat(m, n, 10)
                        .unwrap();
                let b =
                    crate::initializers::ranged_rand_around_mat::<i32>(
                        m, n, 10,
                    )
                    .unwrap();
                let mut a_mut = a.clone();
                let a_p_b = add(&a, &b).unwrap();
                add_inline(&mut a_mut, &b).unwrap();
                assert_eq!(a_mut, a_p_b);

                for i in 0..m {
                    for j in 0..n {
                        let v = a[i][j] + b[i][j];
                        assert_eq!(a_p_b[i][j], v);
                        assert_eq!(a_mut[i][j], v);
                    }
                }
            }
        }
    }
}
