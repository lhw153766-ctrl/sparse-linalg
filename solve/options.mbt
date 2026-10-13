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
/// Reported by every solver, and also useful on its own to check a solution
/// that came from somewhere else.
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
  let r = sub_vec(b, a.spmv(x))
  let nr = norm2(r)
  let nb = norm2(b)
  if nr.is_nan() || nb.is_nan() {
    return 0.0 / 0.0
  }
  if nb == 0.0 {
    // An all-zero right-hand side is solved exactly by x = 0; use the absolute
    // residual so the caller does not get a division by zero.
    return nr
  }
  if nb.is_inf() {
    return 0.0 / 0.0
  }
  nr / nb
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
