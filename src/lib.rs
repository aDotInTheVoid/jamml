extern crate num_traits;
use num_traits::NumAssign;

//mod traits;

/// For now, a matrix is a vector of vectors of numbers
///
/// When I change this, it'll probably be easyer to rewrite from scratch.
/// Although their are probable considerable preformance gains to be made from a cleverer definition.
pub type Mat<T> = Vec<Vec<T>>;

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

/// Calculates the dot product of two Vectors of Numbers.
pub fn dot_product<T: NumAssign + Copy>(a: &Vec<T>, b: &Vec<T>) -> T {
    let mut acc: T = T::zero();
    assert_eq!(a.len(), b.len());
    for (av, bv) in a.iter().zip(b.iter()) {
        acc += (*av) * (*bv);
    }
    return acc;
}

fn isvalid<T>(m: &Mat<T>) -> bool {
    let l = m[0].len();
    for i in m.iter() {
        if i.len() != l {
            return false;
        }
    }
    return true;
}

fn dims<T>(m: &Mat<T>) -> (usize, usize) {
    assert!(isvalid(&m));
    return (m.len(), (m[0]).len());
}

/// Calculates the transpose of a matrix
pub fn transpose<T: NumAssign + Copy>(a: &Mat<T>) -> Mat<T> {
    let (m, n): (usize, usize) = dims(a);
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
    extern crate rand;
    use rand::{thread_rng, Rng};

    #[test]
    fn build_in_integer_aritmatic() {
        assert_eq!(2 + 2, 4);
        assert_eq!(3 * 6, 18);
    }

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
    fn column_vector_initialiser() {
        let v: Mat<i32> = vec_to_column_mat(&vec![1, 2, 3]);
        assert_eq!(v, vec![vec![1], vec![2], vec![3]])
    }

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
