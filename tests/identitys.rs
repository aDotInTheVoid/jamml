extern crate jamml;
extern crate rand;
use rand::Rng;

#[test]
//https://en.wikipedia.org/wiki/Matrix_(mathematics)#Addition,_scalar_multiplication_and_transposition
fn transposition_definition() {
    for _ in 0..10 {
        let a =
            jamml::initializers::ranged_rand_mat(6, 6, -100, 100).unwrap();
        let ta = jamml::ops::transpose(&a).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(ta[i][j], a[j][i]);
            }
        }
    }
}

mod transpose_props {
    use super::*;

    #[test]
    //https://en.wikipedia.org/wiki/Transpose#Properties
    fn double_transpose_is_identity() {
        for _ in 0..10 {
            let a = jamml::initializers::ranged_rand_mat(6, 6, -100, 100)
                .unwrap();
            let tta =
                jamml::ops::transpose(&jamml::ops::transpose(&a).unwrap())
                    .unwrap();

            // A = Att
            assert_eq!(a, tta);
        }
    }

    #[test]
    // https://en.wikipedia.org/wiki/Transpose#Properties
    fn distributivity_wrt_addition() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            // Dims
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);

            // Items
            let a = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let b = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();

            // (A+B)t = (At + Bt)
            assert_eq!(
                jamml::ops::transpose(&jamml::ops::add(&a, &b).unwrap())
                    .unwrap(),
                jamml::ops::add(
                    &jamml::ops::transpose(&b).unwrap(),
                    &jamml::ops::transpose(&a).unwrap()
                )
                .unwrap()
            );
        }
    }

    #[test]
    //https://en.wikipedia.org/wiki/Transpose#Properties
    fn distributivity_wrt_matmul() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            // Dims
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);
            let p: usize = rng.gen_range(1, 20);

            // Items
            let a = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let b = jamml::initializers::ranged_rand_around_mat(n, p, 10)
                .unwrap();

            // (AB)t = (Bt)(At)
            assert_eq!(
                jamml::ops::transpose(
                    &jamml::ops::matmul(&a, &b).unwrap()
                )
                .unwrap(),
                jamml::ops::matmul(
                    &jamml::ops::transpose(&b).unwrap(),
                    &jamml::ops::transpose(&a).unwrap()
                )
                .unwrap()
            );
        }
    }

    #[test]
    //https://en.wikipedia.org/wiki/Transpose#Properties
    fn distributivity_wrt_scalar_mul() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let a = jamml::initializers::ranged_rand_around_mat(6, 6, 100)
                .unwrap();
            let c: i32 = rng.gen_range(-100, 100);

            // scalar then transpose
            let tca = jamml::ops::transpose(
                &jamml::ops::scalar_mul(&a, c).unwrap(),
            )
            .unwrap();

            // transpose then scalar
            let cta = jamml::ops::scalar_mul(
                &jamml::ops::transpose(&a).unwrap(),
                c,
            )
            .unwrap();
            //(cA)t = //c(At)
            assert_eq!(tca, cta);
        }
    }
}

mod matmul_pros {
    use super::*;

    #[test]
    //https://en.wikipedia.org/wiki/Matrix_multiplication#Distributivity
    fn distributivity_wrt_addition() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);
            let p: usize = rng.gen_range(1, 20);
            let q: usize = rng.gen_range(1, 20);

            let a = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let b = jamml::initializers::ranged_rand_around_mat(n, p, 10)
                .unwrap();
            let c = jamml::initializers::ranged_rand_around_mat(n, p, 10)
                .unwrap();
            let d = jamml::initializers::ranged_rand_around_mat(p, q, 10)
                .unwrap();

            // A(B + C) = AB + AC
            assert_eq!(
                jamml::ops::matmul(&a, &jamml::ops::add(&b, &c).unwrap())
                    .unwrap(),
                jamml::ops::add(
                    &jamml::ops::matmul(&a, &b).unwrap(),
                    &jamml::ops::matmul(&a, &c).unwrap()
                )
                .unwrap()
            );

            //(B + C)D = BD + CD
            assert_eq!(
                jamml::ops::matmul(&jamml::ops::add(&b, &c).unwrap(), &d)
                    .unwrap(),
                jamml::ops::add(
                    &jamml::ops::matmul(&b, &d).unwrap(),
                    &jamml::ops::matmul(&c, &d).unwrap()
                )
                .unwrap()
            );
        }
    }

    #[test]
    //https://en.wikipedia.org/wiki/Matrix_multiplication#Product_with_a_scalar
    fn distributivity_wrt_scaler() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            // Dims
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);
            let p: usize = rng.gen_range(1, 20);

            // Items
            let a = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let b = jamml::initializers::ranged_rand_around_mat(n, p, 10)
                .unwrap();
            let c = rng.gen_range(-10, 10);

            // c(AB) = (cA)B
            assert_eq!(
                jamml::ops::scalar_mul(
                    &jamml::ops::matmul(&a, &b).unwrap(),
                    c
                )
                .unwrap(),
                jamml::ops::matmul(
                    &jamml::ops::scalar_mul(&a, c).unwrap(),
                    &b
                )
                .unwrap()
            )
        }
    }

    #[test]
    //https://en.wikipedia.org/wiki/Matrix_multiplication#Transpose
    fn distributivity_wrt_transpose() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            // Dims
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);
            let p: usize = rng.gen_range(1, 20);

            // Items
            let a = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let b = jamml::initializers::ranged_rand_around_mat(n, p, 10)
                .unwrap();

            // (AB)t = (Bt)(At)
            assert_eq!(
                jamml::ops::transpose(
                    &jamml::ops::matmul(&a, &b).unwrap()
                )
                .unwrap(),
                jamml::ops::matmul(
                    &jamml::ops::transpose(&b).unwrap(),
                    &jamml::ops::transpose(&a).unwrap()
                )
                .unwrap()
            );
        }
    }

    #[test]
    //https://en.wikipedia.org/wiki/Matrix_multiplication#Associativity
    fn associativity() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);
            let p: usize = rng.gen_range(1, 20);
            let q: usize = rng.gen_range(1, 20);

            let a = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let b = jamml::initializers::ranged_rand_around_mat(n, p, 10)
                .unwrap();
            let c = jamml::initializers::ranged_rand_around_mat(p, q, 10)
                .unwrap();
            // (AB)C = A(BC)
            assert_eq!(
                jamml::ops::matmul(
                    &jamml::ops::matmul(&a, &b).unwrap(),
                    &c
                )
                .unwrap(),
                jamml::ops::matmul(
                    &a,
                    &jamml::ops::matmul(&b, &c).unwrap()
                )
                .unwrap()
            );
        }
    }
}

mod scalar_mul_props {
    use super::*;
    //https://en.wikipedia.org/wiki/Matrix_(mathematics)#Addition,_scalar_multiplication_and_transposition

    #[test]
    fn scalar_multiplication_definion() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let c: i32 = rng.gen_range(-100, 100);
            let a = jamml::initializers::ranged_rand_mat(6, 6, -100, 100)
                .unwrap();
            let ac = jamml::ops::scalar_mul(&a, c).unwrap();

            for i in 0..6 {
                for j in 0..6 {
                    assert_eq!(ac[i][j], c * a[i][j]);
                }
            }
        }
    }

    #[test]
    fn additive_wrt_scalar() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            // Dims
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);

            // Items
            let v = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let c = rng.gen_range(-10, 10);
            let d = rng.gen_range(-10, 10);

            // (c + d)V =cV + dV
            assert_eq!(
                jamml::ops::scalar_mul(&v, c + d).unwrap(),
                jamml::ops::add(
                    &jamml::ops::scalar_mul(&v, c).unwrap(),
                    &jamml::ops::scalar_mul(&v, d).unwrap()
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn additive_wrt_matrix() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            // Dims
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);

            // Items
            let v = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let w = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let c = rng.gen_range(-10, 10);

            // (c + d)V =cV + dV
            assert_eq!(
                jamml::ops::scalar_mul(
                    &jamml::ops::add(&v, &w).unwrap(),
                    c
                )
                .unwrap(),
                jamml::ops::add(
                    &jamml::ops::scalar_mul(&v, c).unwrap(),
                    &jamml::ops::scalar_mul(&w, c).unwrap()
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn prod_of_scalars() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            // Dims
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);

            // Items
            let v = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();
            let c = rng.gen_range(-10, 10);
            let d = rng.gen_range(-10, 10);

            // (cd)V =c(dV)
            assert_eq!(
                jamml::ops::scalar_mul(&v, c * d).unwrap(),
                jamml::ops::scalar_mul(
                    &jamml::ops::scalar_mul(&v, c).unwrap(),
                    d
                )
                .unwrap()
            );
        }
    }

    #[test]
    fn multiplicative_identity() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            // Dims
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);

            // Items
            let v = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();

            // (1)V = V
            assert_eq!(jamml::ops::scalar_mul(&v, 1).unwrap(), v);
        }
    }
    #[test]
    fn multiplicative_zero_is_void() {
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            // Dims
            let m: usize = rng.gen_range(1, 20);
            let n: usize = rng.gen_range(1, 20);

            // Items
            let v = jamml::initializers::ranged_rand_around_mat(m, n, 10)
                .unwrap();

            // (0)V = 0 (as Mat)
            assert_eq!(
                jamml::ops::scalar_mul(&v, 0).unwrap(),
                jamml::initializers::zero_mat(m, n).unwrap()
            );
        }
    }

}

#[test]
fn dot_product_is_mat_times_mat_transpose() {
    for _ in 0..10 {
        let a =
            jamml::initializers::ranged_rand_around_mat(10, 1, 1).unwrap();
        let b =
            jamml::initializers::ranged_rand_around_mat(10, 1, 1).unwrap();

        let mm =
            jamml::ops::matmul(&a, &jamml::ops::transpose(&b).unwrap())
                .unwrap()[0][0];
        let dp = jamml::ops::dot_product(&b[0], &a[0]).unwrap();

        assert_eq!(mm, dp);
    }
}
