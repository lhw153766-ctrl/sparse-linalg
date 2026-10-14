// Shared types and dense vector helpers for the solver layer.
//
// The storage packages are generic over the scalar type. The solvers are not:
// every one of them needs a convergence test of the form "is this residual
// small enough", which only means something for floating point. Restricting
// this package to `Double` keeps the convergence logic honest and avoids
// pretending that an exact integer solver would terminate the same way.

///|
/// Tuning for an iterative solver.
///
/// `tol` is relative to `||b||`, so it is scale-free: solving a system whose
/// right-hand side is a million times larger does not need a looser tolerance.
pub(all) struct SolverOptions {
  /// Target relative residual, `||b - A x|| / ||b||`.
  tol : Double
  /// Iteration budget. The solver stops and reports `converged: false` when it
  /// is used up.
  max_iter : Int
} derive(Debug, Eq)

///|
/// A tolerance and iteration budget that are sensible for double precision on
/// well-conditioned problems.
pub fn SolverOptions::defaults() -> SolverOptions {
  { tol: 1.0e-10, max_iter: 1000, }
}

///|
/// Build options explicitly.
pub fn SolverOptions::new(
  tol : Double,
  max_iter : Int,
) -> SolverOptions raise @err.SparseError {
  if !(tol > 0.0) || !is_finite_val(tol) {
    raise @err.SparseError::DimensionMismatch(
      "SolverOptions::new: tolerance must be positive and finite, got " +
      tol.to_string(),
    )
  }
  if max_iter <= 0 {
    raise @err.SparseError::DimensionMismatch(
      "SolverOptions::new: max_iter must be positive, got " +
      max_iter.to_string(),
    )
  }
  { tol, max_iter, }
}

///|
/// Validate solver options, rejecting non-positive or non-finite tolerances
/// and non-positive iteration counts.
pub fn check_solver_options(
  options : SolverOptions,
) -> Unit raise @err.SparseError {
  if !(options.tol > 0.0) || !is_finite_val(options.tol) {
    raise @err.SparseError::DimensionMismatch(
      "solver tolerance must be positive and finite, got " +
      options.tol.to_string(),
    )
  }
  if options.max_iter <= 0 {
    raise @err.SparseError::DimensionMismatch(
      "solver max_iter must be positive, got " + options.max_iter.to_string(),
    )
  }
}

///|
/// What a solver produced.
///
/// `converged` is separate from `iterations` on purpose: a caller needs to know
/// whether the answer met the tolerance, not just how long the solver ran. A
/// result with `converged: false` still holds the best iterate reached, which
/// is often good enough to inspect or to restart from.
pub(all) struct SolveResult {
  /// The solution (or the best iterate when the budget ran out).
  x : Array[Double]
  /// Iterations actually performed.
  iterations : Int
  /// Final relative residual, `||b - A x|| / ||b||`.
  residual : Double
  /// Whether `residual <= tol`.
  converged : Bool
} derive(Debug, Eq)

///|
/// Dot product of two dense vectors.
pub fn dot(
  a : Array[Double],
  b : Array[Double],
) -> Double raise @err.SparseError {
  if a.length() != b.length() {
    raise @err.SparseError::DimensionMismatch(
      "dot: lengths differ, " +
      a.length().to_string() +
      " and " +
      b.length().to_string(),
    )
  }
  let mut acc = 0.0
  for i = 0; i < a.length(); i = i + 1 {
    acc = acc + a[i] * b[i]
  }
  acc
}

///|
/// Euclidean norm, scaled to avoid underflow and overflow on extreme vectors.
///
/// Propagates NaN if any entry is NaN. Returns infinity if any entry is infinite
/// (and no entry is NaN).
pub fn norm2(a : Array[Double]) -> Double {
  let mut max_val = 0.0
  let mut has_nan = false
  let mut has_inf = false
  for i = 0; i < a.length(); i = i + 1 {
    let v = a[i]
    if v.is_nan() {
      has_nan = true
    } else if v.is_inf() {
      has_inf = true
    } else {
      let abs_val = Double::abs(v)
      if abs_val > max_val {
        max_val = abs_val
      }
    }
  }
  if has_nan {
    return 0.0 / 0.0
  }
  if has_inf {
    return @double.infinity
  }
  if max_val == 0.0 {
    return 0.0
  }
  let mut acc = 0.0
  for i = 0; i < a.length(); i = i + 1 {
    let scaled = a[i] / max_val
    acc = acc + scaled * scaled
  }
  max_val * Double::sqrt(acc)
}

///|
/// Numerically stable Euclidean length of a 2D vector, `sqrt(a^2 + b^2)`.
pub fn hypot(a : Double, b : Double) -> Double {
  let abs_a = Double::abs(a)
  let abs_b = Double::abs(b)
  if abs_a.is_nan() || abs_b.is_nan() {
    return 0.0 / 0.0
  }
  if abs_a.is_inf() || abs_b.is_inf() {
    return @double.infinity
  }
  if abs_a == 0.0 {
    return abs_b
  }
  if abs_b == 0.0 {
    return abs_a
  }
  let (max_val, min_val) = if abs_a > abs_b {
    (abs_a, abs_b)
  } else {
    (abs_b, abs_a)
  }
  let t = min_val / max_val
  max_val * Double::sqrt(1.0 + t * t)
}

///|
/// True when a scalar value is finite (neither NaN nor infinite).
pub fn is_finite_val(v : Double) -> Bool {
  !v.is_nan() && !v.is_inf()
}

///|
/// True when all elements of the vector are finite (neither NaN nor infinite).
pub fn is_finite_vector(v : Array[Double]) -> Bool {
  for i = 0; i < v.length(); i = i + 1 {
    if v[i].is_nan() || v[i].is_inf() {
      return false
    }
  }
  true
}

///|
/// `y += alpha * x`, in place.
pub fn axpy(
  alpha : Double,
  x : Array[Double],
  y : Array[Double],
) -> Unit raise @err.SparseError {
  if x.length() != y.length() {
    raise @err.SparseError::DimensionMismatch(
      "axpy: lengths differ, " +
      x.length().to_string() +
      " and " +
      y.length().to_string(),
    )
  }
  for i = 0; i < x.length(); i = i + 1 {
    y[i] = y[i] + alpha * x[i]
  }
}

///|
/// `y = alpha * x`, returning a new vector.
pub fn scaled(alpha : Double, x : Array[Double]) -> Array[Double] {
  let out : Array[Double] = Array::make(x.length(), 0.0)
  for i = 0; i < x.length(); i = i + 1 {
    out[i] = alpha * x[i]
  }
  out
}

///|
/// `a - b`, returning a new vector.
pub fn sub_vec(
  a : Array[Double],
  b : Array[Double],
) -> Array[Double] raise @err.SparseError {
  if a.length() != b.length() {
    raise @err.SparseError::DimensionMismatch(
      "sub_vec: lengths differ, " +
      a.length().to_string() +
      " and " +
      b.length().to_string(),
    )
  }
  let out : Array[Double] = Array::make(a.length(), 0.0)
  for i = 0; i < a.length(); i = i + 1 {
    out[i] = a[i] - b[i]
  }
  out
}

///|
/// `a + b`, returning a new vector.
pub fn add_vec(
  a : Array[Double],
  b : Array[Double],
) -> Array[Double] raise @err.SparseError {
  if a.length() != b.length() {
    raise @err.SparseError::DimensionMismatch(
      "add_vec: lengths differ, " +
      a.length().to_string() +
      " and " +
      b.length().to_string(),
    )
  }
  let out : Array[Double] = Array::make(a.length(), 0.0)
  for i = 0; i < a.length(); i = i + 1 {
    out[i] = a[i] + b[i]
  }
  out
}

///|
/// The relative residual `||b - A x|| / ||b||`.
///

///|
/// Scale factor (2^500) used to bring subnormal systems into standard Double
/// normal floating-point range without rounding loss.
pub const SUBNORMAL_SCALE_FACTOR : Double = 3.273390607896142e150

///|
/// Reported by every solver, and also useful on its own to check a solution
/// that came from somewhere else.
/// Each row is evaluated in a power-of-two scaled coordinate system, before
/// taking the norm ratio. In particular, subnormal products are never rounded
/// to zero before subtracting the right-hand side. Rows whose nonzero terms
/// span more than 970 binary orders are rejected rather than certified after
/// silently dropping terms. This is a floating-point residual, not an exact
/// arithmetic certificate for arbitrarily ill-conditioned systems.
pub fn relative_residual(
  a : @csr.Csr[Double],
  x : Array[Double],
  b : Array[Double],
) -> Double raise @err.SparseError {
  if b.length() != a.nrows() {
    raise @err.SparseError::DimensionMismatch(
      "relative_residual: b has length " +
      b.length().to_string() +
      " but the matrix has " +
      a.nrows().to_string() +
      " rows",
    )
  }
  if x.length() != a.ncols() {
    raise @err.SparseError::DimensionMismatch(
      "relative_residual: x length does not match column count",
    )
  }
  let va = a.vals()
  if !is_finite_vector(x) || !is_finite_vector(b) || !is_finite_vector(va) {
    return 0.0 / 0.0
  }
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let n = a.nrows()
  let mantissas : Array[Double] = Array::make(n, 0.0)
  let exponents : Array[Int] = Array::make(n, 0)
  let mut residual_exp = -10000
  let mut rhs_exp = -10000
  for i = 0; i < n; i = i + 1 {
    let mut row_exp = -10000
    if b[i] != 0.0 {
      row_exp = binary_exponent(b[i])
      if row_exp > rhs_exp {
        rhs_exp = row_exp
      }
    }
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      if va[k] != 0.0 && x[ci[k]] != 0.0 {
        let e = binary_exponent(va[k]) + binary_exponent(x[ci[k]]) + 1
        if e > row_exp {
          row_exp = e
        }
      }
    }
    if row_exp == -10000 {
      continue
    }
    let mut sum = 0.0
    let mut correction = 0.0
    if b[i] != 0.0 {
      check_residual_scale(binary_exponent(b[i]) - row_exp)
      sum = @math.scalbn(b[i], -row_exp)
    }
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      let v = va[k]
      let xj = x[ci[k]]
      if v == 0.0 || xj == 0.0 {
        continue
      }
      let ev = binary_exponent(v)
      let ex = binary_exponent(xj)
      check_residual_scale(ev + ex - row_exp)
      let mv = @math.scalbn(v, -ev)
      let mx = @math.scalbn(xj, -ex)
      let product = mv * mx
      // Dekker splitting on normal mantissas also preserves the rounding
      // error of multiplication, which a second ordinary SpMV would repeat.
      let sv = 134217729.0 * mv
      let sx = 134217729.0 * mx
      let hv = sv - (sv - mv)
      let hx = sx - (sx - mx)
      let lv = mv - hv
      let lx = mx - hx
      let product_error = hv * hx - product + hv * lx + lv * hx + lv * lx
      let shift = ev + ex - row_exp
      let term = -@math.scalbn(product, shift)
      let low = -@math.scalbn(product_error, shift)
      if product_error != 0.0 &&
        (low == 0.0 || Double::abs(low) < 2.2250738585072014e-308) {
        raise @err.SparseError::UnsupportedMatrix(
          "relative_residual: product correction exceeds supported precision",
        )
      }
      let (next, error) = compensated_add(sum, term)
      correction = correction + error
      let (next, error) = compensated_add(next, low)
      correction = correction + error
      sum = next
    }
    let r = sum + correction
    if r != 0.0 {
      let er = binary_exponent(r)
      mantissas[i] = @math.scalbn(r, -er)
      exponents[i] = row_exp + er
      if exponents[i] > residual_exp {
        residual_exp = exponents[i]
      }
    }
  }
  if residual_exp == -10000 {
    return 0.0
  }
  for i = 0; i < n; i = i + 1 {
    mantissas[i] = @math.scalbn(mantissas[i], exponents[i] - residual_exp)
  }
  let nr = norm2(mantissas)
  if rhs_exp == -10000 {
    // An all-zero right-hand side uses the absolute residual.
    return @math.scalbn(nr, residual_exp)
  }
  let scaled_b : Array[Double] = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    scaled_b[i] = @math.scalbn(b[i], -rhs_exp)
  }
  @math.scalbn(nr / norm2(scaled_b), residual_exp - rhs_exp)
}

///|
// Neumaier's error term for an addition in the scaled row accumulator.
fn compensated_add(a : Double, b : Double) -> (Double, Double) {
  let sum = a + b
  let error = if Double::abs(a) >= Double::abs(b) {
    a - sum + b
  } else {
    b - sum + a
  }
  (sum, error)
}

///|
// floor(log2(abs(v))) for a finite, nonzero Double, including subnormals.
fn binary_exponent(v : Double) -> Int {
  let bits = Double::abs(v).reinterpret_as_uint64()
  let e = ((bits >> 52) & 2047UL).to_int()
  if e == 0 {
    let normal = @math.scalbn(v, 54)
    ((normal.reinterpret_as_uint64() >> 52) & 2047UL).to_int() - 1023 - 54
  } else {
    e - 1023
  }
}

///|
fn check_residual_scale(exponent : Int) -> Unit raise @err.SparseError {
  if exponent < -970 {
    raise @err.SparseError::UnsupportedMatrix(
      "relative_residual: row dynamic range exceeds supported precision",
    )
  }
}

// Validate that a matrix and right-hand side are compatible, and that the
// system is square. Every solver starts here so the error messages agree.

///|
fn check_system(
  a : @csr.Csr[Double],
  b : Array[Double],
) -> Unit raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  if b.length() != a.nrows() {
    raise @err.SparseError::DimensionMismatch(
      "right-hand side has length " +
      b.length().to_string() +
      " but the matrix is " +
      a.nrows().to_string() +
      "x" +
      a.ncols().to_string(),
    )
  }
  for i = 0; i < b.length(); i = i + 1 {
    let v = b[i]
    if v.is_nan() || v.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "check_system: right-hand side contains non-finite values",
      )
    }
  }
  let va = a.vals()
  for k = 0; k < va.length(); k = k + 1 {
    let v = va[k]
    if v.is_nan() || v.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "check_system: matrix contains non-finite values",
      )
    }
  }
}
