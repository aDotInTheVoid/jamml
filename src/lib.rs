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

//! # JAMML: Just Another Matrix Math Library
//!
//! `core` contains core functionality, like matrix definitions
//!
//! `initializers` contains functionality create matrices
//!
//! `ops` contains functions that operate on matrices

pub mod core;
pub mod initializers;
pub mod ops;

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;

    #[test]
    fn build_in_integer_aritmatic() {
        assert_eq!(2 + 2, 4);
        assert_eq!(3 * 6, 18);
    }

    #[test]
    fn can_see_mods() {
        #[allow(unused_variables)]
        let x: core::Mat<i32> = ops::transpose(&initializers::column_mat(&vec![1, 2, 3]));
    }
}
