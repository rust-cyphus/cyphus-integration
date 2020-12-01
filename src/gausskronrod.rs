//! Module for generating gauss-kronrod nodes and weights for arbitrary rules
//! and arbitrary floating point types. Adapted from the QuadGK.jl Julia package
//! (see (here)[https://github.com/JuliaMath/QuadGK.jl]).

use num::Float;

/// Sort the eigenvalues `d` and eigenvectors `v` in descending order.
fn eigsort<T>(d: &mut [T], v: &mut Vec<Vec<T>>)
where
    T: Float,
{
    let mut k: usize;
    let n = d.len();
    for i in 0..(n - 1) {
        k = i;
        let mut p = d[i];
        for j in i..n {
            if d[j] >= p {
                k = j;
                p = d[j];
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for j in 0..n {
                p = v[j][i];
                v[j][i] = v[j][k];
                v[j][k] = p;
            }
        }
    }
}

/// Rotate two elements of a matrix `m`.
fn rotate<T>(m: &mut Vec<Vec<T>>, s: T, tau: T, i: usize, j: usize, k: usize, l: usize)
where
    T: Float,
{
    let g: T = m[i][j];
    let h: T = m[k][l];
    m[i][j] = g - s * (h + g * tau);
    m[k][l] = h + s * (g - h * tau);
}

/// Compute the eigenvalues and eigenvectors of a symmetric matrix `mmat` using
/// jacobi algorithm.
fn jacobi<T>(mmat: &Vec<Vec<T>>) -> (Vec<T>, Vec<Vec<T>>)
where
    T: Float,
{
    let n = mmat.len();
    // Copy the matrix so we don't destroy original
    let mut mat: Vec<Vec<T>> = (0..n).map(|i| mmat[i].to_owned()).collect();
    let mut v: Vec<Vec<T>> = vec![vec![T::zero(); n]; n];
    // Set b and d to the diagonal of the matrix.
    let mut d: Vec<T> = (0..n).map(|i| mat[i][i]).collect();
    let mut b: Vec<T> = (0..n).map(|i| mat[i][i]).collect();
    let mut z: Vec<T> = vec![T::zero(); n];

    for i in 0..n {
        // Set v to the identity matrix
        v[i][i] = T::one();
    }

    let eps = T::epsilon();
    let max_iter = 100;

    for i in 1..=max_iter {
        let mut sm: T = T::zero();
        for ip in 0..(n - 1) {
            for iq in (ip + 1)..n {
                sm = sm + mat[ip][iq].abs();
            }
        }
        if sm.is_zero() {
            eigsort(&mut d, &mut v);
            return (d, v);
        }
        let thresh = if i < 4 {
            T::from(0.2).unwrap() * sm / T::from(n * n).unwrap()
        } else {
            T::zero()
        };
        for ip in 0..(n - 1) {
            for iq in (ip + 1)..n {
                let g = T::from(100).unwrap() * mat[ip][iq].abs();
                if i > 4 && g <= eps * d[ip].abs() && g <= eps * d[iq].abs() {
                    mat[ip][iq] = T::zero();
                } else if mat[ip][iq].abs() > thresh {
                    let h = d[iq] - d[ip];
                    let t = if g <= eps * h.abs() {
                        mat[ip][iq] / h
                    } else {
                        let theta = T::from(0.5).unwrap() * h / mat[ip][iq];
                        let pre = if theta < T::zero() {
                            -T::one()
                        } else {
                            T::one()
                        };
                        pre / (theta.abs() + (T::one() + theta * theta).sqrt())
                    };
                    let c = T::one() / (T::one() + t * t).sqrt();
                    let s = t * c;
                    let tau = s / (T::one() + c);
                    let h = t * mat[ip][iq];
                    z[ip] = z[ip] - h;
                    z[iq] = z[iq] + h;
                    d[ip] = d[ip] - h;
                    d[iq] = d[iq] + h;
                    mat[ip][iq] = T::zero();
                    for j in 0..ip {
                        rotate(&mut mat, s, tau, j, ip, j, iq);
                    }
                    for j in (ip + 1)..iq {
                        rotate(&mut mat, s, tau, ip, j, j, iq);
                    }
                    for j in (iq + 1)..n {
                        rotate(&mut mat, s, tau, ip, j, iq, j);
                    }
                    for j in 0..n {
                        rotate(&mut v, s, tau, j, ip, j, iq);
                    }
                }
            }
        }
        for ip in 0..n {
            b[ip] = b[ip] + z[ip];
            d[ip] = b[ip];
            z[ip] = T::zero();
        }
    }
    panic!("Maximum number of jacobi iterations reached.");
}

/// Given a symmetric tridiagonal matrix H with H[i,i] = 0 and
/// H[i-1,i] = H[i,i-1] = b[i-1], compute p(z) = det(z I - H) and its
/// derivative p'(z), returning (p,p').
fn eigpoly<T: Float>(b: &[T], z: T, m: usize) -> (T, T) {
    let mut d1 = z;
    let mut d1deriv = T::one();
    let mut d2 = T::one();
    let mut d2deriv = T::zero();
    for i in 1..m {
        let b2 = b[i - 1] * b[i - 1];
        let d = z * d1 - b2 * d2;
        let dderiv = d1 + z * d1deriv - b2 * d2deriv;
        d2 = d1;
        d1 = d;
        d2deriv = d1deriv;
        d1deriv = dderiv;
    }
    (d1, d1deriv)
}

/// compute the n smallest eigenvalues of the symmetric tridiagonal matrix H
/// (defined from b as in eigpoly) using a Newton iteration
/// on det(H - lambda I).  Unlike eig, handles BigFloat.
fn eignewt<T: Float>(b: &[T], m: usize, n: usize) -> Vec<T> {
    let mut h: Vec<Vec<T>> = vec![vec![T::zero(); m]; m];
    for i in 1..m {
        if i != 0 {
            h[i - 1][i] = b[i - 1];
            h[i][i - 1] = b[i - 1];
        }
    }
    // get initial guess from eig on Float64 matrix
    let (lambda0, _) = jacobi(&h);

    let mut lambda = vec![T::zero(); n];
    for i in 0..n {
        lambda[i] = lambda0[i];
        for _ in 0..999 {
            let (p, pderiv) = eigpoly(b, lambda[i], m);
            let lamold = lambda[i];
            lambda[i] = lamold - p / pderiv;
            if (lambda[i] - lamold).abs() < T::from(10).unwrap() * T::epsilon() * lambda[i].abs() {
                break;
            }
        }
        // do one final Newton iteration for luck and profit:
        let (p, pderiv) = eigpoly(b, lambda[i], m);
        lambda[i] = lambda[i] - p / pderiv;
    }
    lambda
}

/// given an eigenvalue z and the matrix H(b) from above, return
/// the corresponding eigenvector, normalized to 1.
fn eigvec1<T>(b: &[T], z: T, m: usize) -> Vec<T>
where
    T: Float,
{
    // "cheat" and use the fact that our eigenvector v must have a
    // nonzero first entries (since it is a quadrature weight), so we
    // can set v[1] = 1 to solve for the rest of the components:.
    let mut v = vec![T::zero(); m];
    v[0] = T::one();
    if m > 1 {
        let mut s: T = v[0];
        v[1] = z * v[0] / b[0];
        s = s + v[1].powi(2);
        for i in 2..m {
            v[i] = -(b[i - 2] * v[i - 2] - z * v[i - 1]) / b[i - 1];
            s = s + v[i].powi(2);
        }
        for el in v.iter_mut() {
            *el = *el * s.sqrt().recip();
        }
    }
    v
}

/// Return a pair `(x, w)` of `n` quadrature points `x` and weights `w` to
/// integrate functions on the interval `(a, b)` .  Uses the method described
/// in Trefethen & Bau, Numerical Linear Algebra, to find the `N`-point Gaussian
/// quadrature in O(`n`^2) operations.
pub fn gauss<T>(n: usize) -> (Vec<T>, Vec<T>)
where
    T: Float,
{
    if n < 1 {
        panic!("Gauss rules require positive order.");
    }
    let o = T::one();

    let b: Vec<T> = (1..=(n - 1))
        .map(|i| T::from(i).unwrap() / (T::from(4 * i * i).unwrap() - o).sqrt())
        .collect::<Vec<T>>();

    let x = eignewt(&b, n, n);
    let w = (0..n)
        .map(|i| T::from(2).unwrap() * eigvec1(&b, x[i], b.len() + 1)[0].powi(2))
        .collect::<Vec<T>>();
    (x, w)
}

/// Compute `2n+1` Kronrod points `x` and weights `w` based on the description in
/// Laurie (1997), appendix A, simplified for `a=0`, for integrating on `[-1,1]`.
/// Since the rule is symmetric, this only returns the `n+1` points with `x >= 0`.
/// The function also computes the embedded `n`-point Gauss quadrature weights `gw`
/// (again for `x >= 0`). Returns `(x,w,wg)` in O(`n`^2) operations. Use `qk` to
/// integrate a function with the produced nodes and weights.
///
/// # Examples
///
/// Generate the 15-pt Gauss-Kronrod rule and integrate e^x (note - using
/// `fixed_order_gauss_kronrod` is faster in this case since 15-pt rule
/// is availible for f64):
/// ```rust
/// use cyphus_integration::prelude::{qk, kronrod};
/// let (nodes, wts_kron, wts_gauss) = kronrod::<f64>(7);
/// let f = |x:f64| {x.exp()};
/// let (res, err, _, _) = qk(f, 0.0, 1.0, &nodes, &wts_kron, &wts_gauss);
/// assert!((res - 1f64.exp() + 1.0).abs() < err);
/// ```
/// This also works with `f32` (and any type the implements `num::Float`):
/// ```
/// use cyphus_integration::prelude::{kronrod, qk};
/// let (nodes, wts_kron, wts_gauss) = kronrod::<f32>(7);
/// let f = |x: f32| x.exp();
/// let (res, err, _, _) = qk(f, 0.0, 1.0, &nodes, &wts_kron, &wts_gauss);
/// assert!((res - 1f32.exp() + 1.0).abs() < err);
/// ```
pub fn kronrod<T>(n: usize) -> (Vec<T>, Vec<T>, Vec<T>)
where
    T: Float,
{
    if n == 0 {
        panic!("Gauss-Kronrod rules require non-zero order.");
    }
    let o = T::one();
    let mut b = vec![T::zero(); 2 * n + 1];
    b[0] = T::from(2).unwrap() * o;
    for j in 1..=((3 * n + 1) / 2) {
        b[j] = T::from(j * j).unwrap() / (T::from(4 * j * j).unwrap() - o);
    }
    let mut s = vec![T::zero(); n / 2 + 2];
    let mut t = vec![T::zero(); n / 2 + 2];
    t[1] = b[n + 1];
    for m in 0..=(n - 2) {
        let mut u = T::zero();
        for k in (0..=((m + 1) / 2)).rev() {
            let l = m - k + 1;
            let k1 = k + n + 2;
            u = u + b[k1 - 1] * s[k] - b[l - 1] * s[k + 1];
            s[k + 1] = u;
        }
        s.swap_with_slice(&mut t);
    }
    for j in (0..=(n / 2)).rev() {
        s[j + 1] = s[j]
    }
    for m in (n - 1)..=(2 * n - 3) {
        let mut u = T::zero();
        for k in (m + 1 - n)..=((m - 1) / 2) {
            let l = m - k + 1;
            let j = n - l;
            let k1 = k + n + 2;
            u = u - (b[k1 - 1] * s[j + 1] - b[l - 1] * s[j + 2]);
            s[j + 1] = u;
        }
        let k = (m + 1) / 2;
        if 2 * k != m {
            let j = n - (m - k + 2);
            b[k + n + 1] = s[j + 1] / s[j + 2];
        }
        s.swap_with_slice(&mut t);
    }
    for j in 1..=(2 * n) {
        b[j - 1] = b[j].sqrt();
    }

    // get negative quadrature points x
    let x = eignewt(&b, 2 * n + 1, n + 1); // x <= 0

    // get quadrature weights
    let w = (1..=(n + 1))
        .map(|i| T::from(2).unwrap() * eigvec1(&b, x[i - 1], 2 * n + 1)[0].powi(2))
        .collect::<Vec<T>>();

    // Get embedded Gauss rule from even-indexed points, using
    // the method described in Trefethen and Bau.
    for j in 1..=(n - 1) {
        b[j - 1] = T::from(j).unwrap() / (T::from(4 * j * j).unwrap() - o).sqrt();
    }
    let gw = (2..=(n + 1))
        .step_by(2)
        .map(|i| T::from(2).unwrap() * eigvec1(&b, x[i - 1], n)[0].powi(2))
        .collect::<Vec<T>>();

    (x, w, gw)
}

#[cfg(test)]
mod test {
    use super::{gauss, jacobi, kronrod};
    #[test]
    fn test_jacobi() {
        let m: Vec<Vec<f64>> = vec![
            vec![1.0602802, 0.50964708, 0.84640266, 1.31369664, 0.08029011],
            vec![0.50964708, 0.20615608, 0.80307108, 0.38519561, 0.6696847],
            vec![0.84640266, 0.80307108, 0.8537295, 1.15432095, 1.46714288],
            vec![1.31369664, 0.38519561, 1.15432095, 1.68956949, 1.13304321],
            vec![0.08029011, 0.6696847, 1.46714288, 1.13304321, 0.02104539],
        ];

        let evals = vec![4.41631788, 0.86626791, 0.27252966, -0.43017322, -1.29416156];
        let (evals_jac, _) = jacobi(&m);
        println!("{:?}", &evals);
        println!("{:?}", &evals_jac);

        for (e, ejac) in evals.iter().zip(evals_jac.iter()) {
            let frac_diff = ((*e - *ejac) / *e).abs();
            assert!(frac_diff < 1e-7);
        }
    }

    #[test]
    fn test_gauss() {
        let xs = [
            0.949107912342759,
            0.741531185599394,
            0.405845151377397,
            0.000000000000000,
            -0.405845151377397,
            -0.741531185599394,
            -0.949107912342759,
        ];
        let ws = [
            0.129484966168870,
            0.279705391489277,
            0.381830050505119,
            0.417959183673469,
            0.381830050505119,
            0.279705391489277,
            0.129484966168870,
        ];

        let (x, w) = gauss::<f64>(7);

        for i in 0..7 {
            let diff1 = (x[i] - xs[i]).abs();
            let diff2 = (w[i] - ws[i]).abs();
            assert!(diff1 < 1e-15);
            assert!(diff2 < 1e-15);
        }
    }
    #[test]
    fn test_kronrod() {
        let ns_test = [
            0.991455371120813,
            0.949107912342759,
            0.864864423359769,
            0.741531185599394,
            0.586087235467691,
            0.405845151377397,
            0.207784955007898,
            0.000000000000000,
        ];
        let gws_test = [
            0.129484966168870,
            0.279705391489277,
            0.381830050505119,
            0.417959183673469,
        ];
        let kws_test = [
            0.022935322010529,
            0.063092092629979,
            0.104790010322250,
            0.140653259715525,
            0.169004726639267,
            0.190350578064785,
            0.204432940075298,
            0.209482141084728,
        ];
        let (ns, kws, gws) = kronrod::<f64>(7);

        for i in 0..8 {
            assert!((ns[i] - ns_test[i]).abs() < 1e-15);
            assert!((kws[i] - kws_test[i]).abs() < 1e-15);
        }
        for i in 0..4 {
            assert!((gws[i] - gws_test[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_kronrod_61() {
        let XGK61: [f64; 31] = [
            0.999_484_410_050_490_637_571_325_895_705_811,
            0.996_893_484_074_649_540_271_630_050_918_695,
            0.991_630_996_870_404_594_858_628_366_109_486,
            0.983_668_123_279_747_209_970_032_581_605_663,
            0.973_116_322_501_126_268_374_693_868_423_707,
            0.960_021_864_968_307_512_216_871_025_581_798,
            0.944_374_444_748_559_979_415_831_324_037_439,
            0.926_200_047_429_274_325_879_324_277_080_474,
            0.905_573_307_699_907_798_546_522_558_925_958,
            0.882_560_535_792_052_681_543_116_462_530_226,
            0.857_205_233_546_061_098_958_658_510_658_944,
            0.829_565_762_382_768_397_442_898_119_732_502,
            0.799_727_835_821_839_083_013_668_942_322_683,
            0.767_777_432_104_826_194_917_977_340_974_503,
            0.733_790_062_453_226_804_726_171_131_369_528,
            0.697_850_494_793_315_796_932_292_388_026_640,
            0.660_061_064_126_626_961_370_053_668_149_271,
            0.620_526_182_989_242_861_140_477_556_431_189,
            0.579_345_235_826_361_691_756_024_932_172_540,
            0.536_624_148_142_019_899_264_169_793_311_073,
            0.492_480_467_861_778_574_993_693_061_207_709,
            0.447_033_769_538_089_176_780_609_900_322_854,
            0.400_401_254_830_394_392_535_476_211_542_661,
            0.352_704_725_530_878_113_471_037_207_089_374,
            0.304_073_202_273_625_077_372_677_107_199_257,
            0.254_636_926_167_889_846_439_805_129_817_805,
            0.204_525_116_682_309_891_438_957_671_002_025,
            0.153_869_913_608_583_546_963_794_672_743_256,
            0.102_806_937_966_737_030_147_096_751_318_001,
            0.051_471_842_555_317_695_833_025_213_166_723,
            0.000_000_000_000_000_000_000_000_000_000_000,
        ];
        let WG61: [f64; 15] = [
            0.007_968_192_496_166_605_615_465_883_474_674,
            0.018_466_468_311_090_959_142_302_131_912_047,
            0.028_784_707_883_323_369_349_719_179_611_292,
            0.038_799_192_569_627_049_596_801_936_446_348,
            0.048_402_672_830_594_052_902_938_140_422_808,
            0.057_493_156_217_619_066_481_721_689_402_056,
            0.065_974_229_882_180_495_128_128_515_115_962,
            0.073_755_974_737_705_206_268_243_850_022_191,
            0.080_755_895_229_420_215_354_694_938_460_530,
            0.086_899_787_201_082_979_802_387_530_715_126,
            0.092_122_522_237_786_128_717_632_707_087_619,
            0.096_368_737_174_644_259_639_468_626_351_810,
            0.099_593_420_586_795_267_062_780_282_103_569,
            0.101_762_389_748_405_504_596_428_952_168_554,
            0.102_852_652_893_558_840_341_285_636_705_415,
        ];
        let WGK61: [f64; 31] = [
            0.001_389_013_698_677_007_624_551_591_226_760,
            0.003_890_461_127_099_884_051_267_201_844_516,
            0.006_630_703_915_931_292_173_319_826_369_750,
            0.009_273_279_659_517_763_428_441_146_892_024,
            0.011_823_015_253_496_341_742_232_898_853_251,
            0.014_369_729_507_045_804_812_451_432_443_580,
            0.016_920_889_189_053_272_627_572_289_420_322,
            0.019_414_141_193_942_381_173_408_951_050_128,
            0.021_828_035_821_609_192_297_167_485_738_339,
            0.024_191_162_078_080_601_365_686_370_725_232,
            0.026_509_954_882_333_101_610_601_709_335_075,
            0.028_754_048_765_041_292_843_978_785_354_334,
            0.030_907_257_562_387_762_472_884_252_943_092,
            0.032_981_447_057_483_726_031_814_191_016_854,
            0.034_979_338_028_060_024_137_499_670_731_468,
            0.036_882_364_651_821_229_223_911_065_617_136,
            0.038_678_945_624_727_592_950_348_651_532_281,
            0.040_374_538_951_535_959_111_995_279_752_468,
            0.041_969_810_215_164_246_147_147_541_285_970,
            0.043_452_539_701_356_069_316_831_728_117_073,
            0.044_814_800_133_162_663_192_355_551_616_723,
            0.046_059_238_271_006_988_116_271_735_559_374,
            0.047_185_546_569_299_153_945_261_478_181_099,
            0.048_185_861_757_087_129_140_779_492_298_305,
            0.049_055_434_555_029_778_887_528_165_367_238,
            0.049_795_683_427_074_206_357_811_569_379_942,
            0.050_405_921_402_782_346_840_893_085_653_585,
            0.050_881_795_898_749_606_492_297_473_049_805,
            0.051_221_547_849_258_772_170_656_282_604_944,
            0.051_426_128_537_459_025_933_862_879_215_781,
            0.051_494_729_429_451_567_558_340_433_647_099,
        ];

        let (ns, kws, gws) = kronrod::<f64>(30);

        for i in 0..31 {
            assert!(dbg!((ns[i] - XGK61[i]).abs()) < 1e-15);
            assert!(dbg!((kws[i] - WGK61[i]).abs()) < 1e-15);
        }
        for i in 0..15 {
            assert!(dbg!((gws[i] - WG61[i]).abs()) < 1e-15);
        }
    }
    #[test]
    fn test_qk_kronrod() {
        use crate::prelude::{kronrod, qk};
        let (nodes, wts_kron, wts_gauss) = kronrod::<f64>(7);
        let f = |x: f64| x.exp();
        let (res, err, _, _) = qk(f, 0.0, 1.0, &nodes, &wts_kron, &wts_gauss);
        assert!((res - 1f64.exp() + 1.0).abs() < err);
    }
    #[test]
    fn test_qk_kronrod_f32() {
        use crate::prelude::{kronrod, qk};
        let (nodes, wts_kron, wts_gauss) = kronrod::<f32>(7);
        let f = |x: f32| x.exp();
        let (res, err, _, _) = qk(f, 0.0, 1.0, &nodes, &wts_kron, &wts_gauss);
        assert!((res - 1f32.exp() + 1.0).abs() < err);
    }
}
