extern crate num;

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
pub fn vec_to_column_mat<T: num::Num>(v: Vec<T>) -> Mat<T> {
    let mut m: Mat<T> = Vec::new();

    for i in v {
        m.push(vec![i]);
    }
    return m;
}

///Calculates the dot product of two Vectors of Numbers.
pub fn dot_product<T>(a: Vec<T>, b: Vec<T>) -> T
where
    // TODO: Cleanup. What i realy nead is for the `num` crate to get their shit together
    // so num::Num includes shit like std::ops::Mul. Until then, I may need to make an alias.
    T: std::ops::AddAssign,
    T: num::Num,
    T: std::ops::Mul,
    T: Copy,
{
    let mut acc: T = T::zero();
    for (av, bv) in a.iter().zip(b.iter()) {
        acc += (*av) * (*bv);
    }
    return acc;
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let v: Mat<i32> = vec_to_column_mat(vec![1, 2, 3]);
        assert_eq!(v, vec![vec![1], vec![2], vec![3]])
    }

    #[test]
    fn dot_product_2_elem() {
        assert_eq!(dot_product(vec![2, 5], vec![3, 1]), 11);
        assert_eq!(dot_product(vec![4, 3], vec![3, 5]), 27);
    }

    #[test]
    fn dot_product_3_elem() {
        assert_eq!(dot_product(vec![1, 3, -5], vec![4, -2, -1]), 3);
        assert_eq!(dot_product(vec![3, 1, 8], vec![4, 2, 3]), 38);
        assert_eq!(dot_product(vec![2, 5, -2], vec![1, 8, -3]), 48);
    }
}
