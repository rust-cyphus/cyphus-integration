# cyphus-integration
Rust library for numerical integration based on GSL.

## Usage

### Builder
The easiest way to use cyphus-integration is through the builder pattern. To
construct an integrator, use:
```rust
use cyphus_integration::prelude::*;
let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(0.0)
            .epsrel(1e-3)
            .key(1)
            .build();
let f = |x:f64| x.ln() / (1.0 + 100.0 * x * x);
let result = gk.integrate(f, 0.0, f64::INFINITY);
let analytic = -0.36168922062077324062;
assert!((result.val - analytic).abs() < 1e-3);
```
In the above, we are setting the absolute tolerance to `0`, the relative tolerance to `1e-3`, the "key" to `1` (this instructs the integrator to use the 15pt Gauss-Krondrod rule) and then we build. To integrate a function, we simply pass the function, the lower bound and the upper bound. If one of the bounds should be infinite, use `f64::INFINITY` or `f64::NEG_INFINITY`.

This builder can also construct a integrator which will handle singularies. For example, suppose we want to integrate `f(x)=x^3 ln(|(x^2-1)(x^2-2)|)`. This function is singular at `x=1` and `x=sqrt(2)`. To let the integrator know this, we use:
```rust
use cyphus_integration::prelude::*;
let gk = GaussKronrodIntegratorBuilder::default()
            .epsabs(0.0)
            .epsrel(1e-3)
            .key(1)
            .singular_points(vec![1f64, 2f64.sqrt()])
            .build();
let f = |x:f64| x.powi(3) * ((x*x-1.0) * (x*x-2.0)).abs().ln();
let result = gk.integrate(f, 0.0, 3.0);
let analytic = 52.740748383471444993;
assert!((result.val - analytic).abs() < 1e-3);
```

### Using `GaussKronrodIntegrator`
You can also just use the `GaussKronrodIntegrator` directly:
```rust
use cyphus_integration::prelude::*;
let gk = GaussKronrodIntegrator{
            epsabs: 0.0,
            epsrel: 1e-3,
            key: 1,
            singular_points: vec![1.0, 2f64.sqrt()],
            limit: 1000,
        };
let f = |x:f64| x.powi(3) * ((x*x-1.0) * (x*x-2.0)).abs().ln();
let result = gk.integrate(f, 0.0, 3.0);
let analytic = 52.740748383471444993;
assert!((result.val - analytic).abs() < 1e-3);
```

### Lower-level function
We also supply access to the lower level function: `qag`, `qags`, `qagp` and `qagi`:
```rust
use cyphus::integration::qagi;

let f = |x:f64| x.ln() / (1.0 + 100.0 * x * x);
// Signiture: func, a, b, epsabs, epsrel, limit, key
let result = qagi(f, 0.0, f64::INFINITY, 0.0, 1e-3, 1000, 1);
let analytic = -0.36168922062077324062;
assert!((result.val - analytic).abs() < 1e-3);
```
