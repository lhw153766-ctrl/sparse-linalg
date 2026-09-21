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
  if !(tol > 0.0) {
    raise @err.SparseError::DimensionMismatch(
      "SolverOptions::new: tolerance must be positive, got " + tol.to_string(),
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
/// Euclidean norm.
pub fn norm2(a : Array[Double]) -> Double {
  let mut acc = 0.0
  for i = 0; i < a.length(); i = i + 1 {
    acc = acc + a[i] * a[i]
  }
  Double::sqrt(acc)
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
  let nb = norm2(b)
  if nb == 0.0 {
    // An all-zero right-hand side is solved exactly by x = 0; use the absolute
    // residual so the caller does not get a division by zero.
    return norm2(r)
  }
  norm2(r) / nb
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
}
