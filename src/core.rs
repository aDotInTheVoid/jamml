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

const MAT_INVALID_ERR_STR: &str = "Matrix invalid (not rectangular)";

/// For now, a matrix is a vector of vectors of numbers
///
/// When I change this, it'll probably be easyer to rewrite from scratch.
/// Although their are probable considerable preformance gains to be made from a cleverer definition.
pub type Mat<T /* Copy + NumAssign*/> = Vec<Vec<T>>;

/// Finds weather or not each row in a matrix it the same lenght.
pub fn isvalid<T: Copy + NumAssign>(m: &Mat<T>) -> bool {
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
/// Panicks if the matrix isn't valid
pub fn dims<T: NumAssign + Copy>(m: &Mat<T>) -> (usize, usize) {
    assert!(isvalid(&m), MAT_INVALID_ERR_STR);
    return (m.len(), (m[0]).len());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matrix_type() {
        #[allow(unused_variables)]
        let x: Mat<i32> = vec![vec![1, 20000, 3], vec![4, -5, 6], vec![-7, -8, 9]];
        #[allow(unused_variables)]
        let x: Mat<f32> = vec![
            vec![10.6, 2.0, 3.3],
            vec![4.6, 5.3, 6.0],
            vec![-7.64, 8.2435, 9.2435],
        ];
        #[allow(unused_variables)]
        let x: Mat<u32> = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    }

    #[test]
    fn dimensions_works() {
        let x = vec![vec![1, 2, 3, 4], vec![4, 5, 4, 0]];
        assert_eq!(dims(&x), (2, 4))
    }

    #[test]
    #[should_panic]
    fn dims_panics_on_invalid() {
        dims(&vec![vec![1, 2, 3], vec![1]]);
    }

    #[test]
    fn isvalid_works() {
        assert!(isvalid(&vec![vec![1, 2, 3, 4], vec![4, 5, 4, 0]]));
        assert!(!isvalid(&vec![vec![1, 2, 3], vec![1]]))
    }

}
