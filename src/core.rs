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

//! Core Matrix Deffinions and auxilarry functions.
//!
//! ```
//! # use jamml::core::*;
//! let m: Mat<i32> = vec![vec![1, 2, 3],
//!                        vec![4, 5, 6],
//!                        vec![7, 8, 9]];
//!
//! assert!(isvalid(&m));
//! assert_eq!(dims(&m).unwrap(), (3, 3));

extern crate num_traits;
use num_traits::NumAssign;

/// A matrix is a vector of vectors of numbers
///
/// ```
/// # use jamml::core::Mat;
/// let m: Mat<i32> = vec![vec![1, 2, 3],
///                        vec![4, 5, 6],
///                        vec![7, 8, 9]];
/// ```
// When I change this, it'll probably be easier to rewrite the whole librafrom scratch.
// Although their are probable considerable preformance gains to be made from a cleverer definition.
pub type Mat<T /* Copy + NumAssign*/> = Vec<Vec<T>>;

/// Finds weather or not each row in a matrix it the same lenght.
///
/// ```
/// # use jamml::core::isvalid;
/// let x = vec![vec![1, 2, 3, 4],
///              vec![5, 6, 7, 8]];
///
/// let y = vec![vec![1, 2, 3],
///              vec![5, 6, 7, 8]];
///
/// assert_eq!(isvalid(&x), true);
/// assert_eq!(isvalid(&y), false);
/// ```
pub fn isvalid<T>(m: &Mat<T>) -> bool
where
    T: Copy + NumAssign,
{
    let l = m[0].len();
    for i in m.iter() {
        if i.len() != l {
            return false;
        }
    }
    return true;
}

/// Return the dimensions of a matrix.
///
/// Returns `Ok(usize, usize)` if the `Vec` of `Vec` is square. Otherwise Returns `MatrixError::NotRectangle`
///
/// ```
/// # use jamml::core::{MatrixError, dims};
/// let x = vec![vec![1, 2, 3, 4],
///              vec![5, 6, 7, 8]];
///
/// let y = vec![vec![1, 2, 3],
///              vec![5, 6, 7, 8]];
///
/// assert_eq!(dims(&x).unwrap(), (2, 4));
/// assert_eq!(dims(&y).err(), Some(MatrixError::NotRectangle));
/// ```
pub fn dims<T>(m: &Mat<T>) -> Result<(usize, usize), MatrixError>
where
    T: NumAssign + Copy,
{
    if isvalid(&m) {
        return Ok((m.len(), (m[0]).len()));
    } else {
        return Err(MatrixError::NotRectangle);
    }
}

/// One of jammls errors
#[derive(Debug, PartialEq)]
pub enum MatrixError {
    /// The vector of vectors were not all even lenght.
    NotRectangle,
    /// The matrix suplied had invalid dimenstions or the dimentions suplyed to create a matrix were invalid
    InvalidDims,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_type() {
        #[allow(unused_variables)]
        let x: Mat<i32> =
            vec![vec![1, 20000, 3], vec![4, -5, 6], vec![-7, -8, 9]];
        #[allow(unused_variables)]
        let x: Mat<f32> = vec![
            vec![10.6, 2.0, 3.3],
            vec![4.6, 5.3, 6.0],
            vec![-7.64, 8.2435, 9.2435],
        ];
        #[allow(unused_variables)]
        let x: Mat<u32> =
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    }

    #[test]
    fn dimensions_works() {
        let x = vec![vec![1, 2, 3, 4], vec![4, 5, 4, 0]];
        assert_eq!(dims(&x).unwrap(), (2, 4))
    }

    #[test]
    fn dims_errors_on_invalid() {
        assert!(dims(&vec![vec![1, 2, 3], vec![1]]).is_err());
    }

    #[test]
    fn isvalid_works() {
        assert!(isvalid(&vec![vec![1, 2, 3, 4], vec![4, 5, 4, 0]]));
        assert!(!isvalid(&vec![vec![1, 2, 3], vec![1]]))
    }

}
