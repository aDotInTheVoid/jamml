extern crate jamml;
extern crate rand;
use rand::Rng;


#[test]
fn scalar_multiplication_definion(){
    let mut rng = rand::thread_rng();
    for _ in 0..10 {
        let c: i32 = rng.gen_range(-100, 100);
        let a = jamml::initializers::ranged_rand_mat(6, 6, -100, 100);
        let ac = jamml::ops::scalar_mul(&a, c);

        for i in 0..6{
            for j in 0..6{
                assert_eq!(ac[i][j], c * a[i][j]);
            }
        }
    }
}

#[test]
fn transposition_definition(){
    for _ in 0..10 {
        let a = jamml::initializers::ranged_rand_mat(6, 6, -100, 100);
        let ta = jamml::ops::transpose(&a);
        for i in 0..6{
            for j in 0..6{
                assert_eq!(ta[i][j], a[j][i]);
            }
        }
    }
}

#[test]
fn double_transpose_is_identity(){
    for _ in 0..10 {
        let a = jamml::initializers::ranged_rand_mat(6, 6, -100, 100);
        let tta = jamml::ops::transpose(&jamml::ops::transpose(&a));
        assert_eq!(a, tta);
    }
}

#[test]
fn transpose_scalar_is_scalar_transpose (){
    let mut rng = rand::thread_rng();
    for _ in 0..10 {
        let a = jamml::initializers::ranged_rand_mat(6, 6, -100, 100);
        let c: i32 = rng.gen_range(-100, 100);

        // scalar then transpose
        let tca = jamml::ops::transpose(&jamml::ops::scalar_mul(&a, c));

        // transpose then scalar
        let cta = jamml::ops::scalar_mul(&jamml::ops::transpose(&a), c);
        assert_eq!(tca, cta);
    }
}