// Preconditioners.
//
// A Krylov method converges at a rate set by the spectrum of the matrix, which
// for a discretised differential operator gets worse as the mesh is refined.
// A preconditioner replaces the system `A x = b` with `M^-1 A x = M^-1 b` where
// `M` approximates `A` but is cheap to invert. The point is not to make each
// iteration cheaper — it is to make far fewer iterations necessary.
//
// Two are provided. Jacobi is the diagonal of `A`, which costs one division per
// entry and helps when the matrix is badly scaled. ILU(0) is an incomplete
// factorisation that keeps exactly the sparsity pattern of `A`, which is much
// stronger on the elliptic problems that motivate sparse solvers in the first
// place.

///|
/// A diagonal (Jacobi) preconditioner: `M = diag(A)`.
pub struct JacobiPrecond {
  inv_diag : Array[Double]
} derive(Debug)

///|
/// Build the Jacobi preconditioner, rejecting a matrix with a zero on the
/// diagonal since the preconditioner would not be invertible.
pub fn JacobiPrecond::new(
  a : @csr.Csr[Double],
) -> JacobiPrecond raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  let d = a.diagonal()
  let inv : Array[Double] = Array::make(d.length(), 0.0)
  for i = 0; i < d.length(); i = i + 1 {
    if d[i] == 0.0 {
      raise @err.SparseError::SingularPivot(i, d[i])
    }
    inv[i] = 1.0 / d[i]
  }
  { inv_diag: inv, }
}

///|
/// `z = M^-1 r`, which for Jacobi is a componentwise division.
pub fn JacobiPrecond::apply(
  self : JacobiPrecond,
  r : Array[Double],
) -> Array[Double] raise @err.SparseError {
  if r.length() != self.inv_diag.length() {
    raise @err.SparseError::DimensionMismatch(
      "JacobiPrecond::apply: r has length " +
      r.length().to_string() +
      ", expected " +
      self.inv_diag.length().to_string(),
    )
  }
  let out : Array[Double] = Array::make(r.length(), 0.0)
  for i = 0; i < r.length(); i = i + 1 {
    out[i] = r[i] * self.inv_diag[i]
  }
  out
}

///|
/// The diagonal used by the preconditioner, for inspection and testing.
pub fn JacobiPrecond::inv_diag(self : JacobiPrecond) -> Array[Double] {
  self.inv_diag
}

///|
/// An incomplete LU factorisation with zero fill: `A ~ L U`, where `L` and `U`
/// share the sparsity pattern of `A` exactly.
///
/// The factors are stored in a single CSR structure. The strictly lower part
/// holds the multipliers of `L` (whose diagonal is implicitly 1) and the upper
/// part including the diagonal holds `U`.
pub struct Ilu0Precond {
  n : Int
  row_ptr : Array[Int]
  col_idx : Array[Int]
  vals : Array[Double]
  // Position of each diagonal entry, so the triangular solves do not have to
  // search for it on every iteration.
  diag_pos : Array[Int]
} derive(Debug)

///|
/// Positions of the diagonal in each row, or `None` when some row has no
/// diagonal entry. ILU(0) is only defined when the diagonal is present.
fn find_diagonals(a : @csr.Csr[Double]) -> Array[Int]? {
  let n = a.nrows()
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let pos : Array[Int] = Array::make(n, -1)
  for i = 0; i < n; i = i + 1 {
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      if ci[k] == i {
        pos[i] = k
      }
    }
    if pos[i] < 0 {
      return None
    }
  }
  Some(pos)
}

// Position of column `j` inside row `i` of a CSR pattern, or -1.

///|
fn find_in_row(rp : Array[Int], ci : Array[Int], i : Int, j : Int) -> Int {
  let mut lo = rp[i]
  let mut hi = rp[i + 1]
  while lo < hi {
    let mid = lo + (hi - lo) / 2
    let c = ci[mid]
    if c == j {
      return mid
    } else if c < j {
      lo = mid + 1
    } else {
      hi = mid
    }
  }
  -1
}

///|
/// Compute the ILU(0) factorisation.
///
/// Rows are processed in increasing order and each row is updated in place.
/// When row `i` is being eliminated, the rows it reads (`k < i`) are already
/// final, so no separate factor storage is needed and the update is a single
/// pass over the row's own entries.
pub fn Ilu0Precond::new(
  a : @csr.Csr[Double],
) -> Ilu0Precond raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  let n = a.nrows()
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let pos = match find_diagonals(a) {
    Some(p) => p
    None =>
      raise @err.SparseError::UnsupportedMatrix(
        "Ilu0Precond::new: a row has no diagonal entry, so the zero-fill factorisation is undefined",
      )
  }
  let vals = a.vals().copy()
  for i = 0; i < n; i = i + 1 {
    for kk = rp[i]; kk < rp[i + 1]; kk = kk + 1 {
      let k = ci[kk]
      if k >= i {
        // Entries at or above the diagonal belong to U and are not eliminated.
        break
      }
      let pivot = vals[pos[k]]
      if pivot == 0.0 {
        raise @err.SparseError::SingularPivot(k, pivot)
      }
      vals[kk] = vals[kk] / pivot
      let lik = vals[kk]
      // Only columns that already exist in row i may be updated; that is what
      // "zero fill" means.
      for jj = kk + 1; jj < rp[i + 1]; jj = jj + 1 {
        let kj = find_in_row(rp, ci, k, ci[jj])
        if kj >= 0 {
          vals[jj] = vals[jj] - lik * vals[kj]
        }
      }
    }
  }
  { n, row_ptr: rp, col_idx: ci, vals, diag_pos: pos, }
}

///|
/// Number of stored entries in the factors, which equals the number in `A`.
pub fn Ilu0Precond::nnz(self : Ilu0Precond) -> Int {
  self.vals.length()
}

///|
/// The factor values, for inspection and testing.
pub fn Ilu0Precond::vals(self : Ilu0Precond) -> Array[Double] {
  self.vals
}

///|
/// `z = M^-1 r`, computed as one forward solve with `L` and one backward solve
/// with `U`.
pub fn Ilu0Precond::apply(
  self : Ilu0Precond,
  r : Array[Double],
) -> Array[Double] raise @err.SparseError {
  if r.length() != self.n {
    raise @err.SparseError::DimensionMismatch(
      "Ilu0Precond::apply: r has length " +
      r.length().to_string() +
      ", expected " +
      self.n.to_string(),
    )
  }
  let n = self.n
  let rp = self.row_ptr
  let ci = self.col_idx
  let va = self.vals
  // Forward solve L y = r. L is unit lower triangular, so no division here.
  let y : Array[Double] = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    let mut acc = r[i]
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      let j = ci[k]
      if j >= i {
        break
      }
      acc = acc - va[k] * y[j]
    }
    y[i] = acc
  }
  // Backward solve U z = y.
  let z : Array[Double] = Array::make(n, 0.0)
  for ii = 0; ii < n; ii = ii + 1 {
    let i = n - 1 - ii
    let mut acc = y[i]
    for k = rp[i + 1] - 1; k >= rp[i]; k = k - 1 {
      let j = ci[k]
      if j <= i {
        break
      }
      acc = acc - va[k] * z[j]
    }
    let pivot = va[self.diag_pos[i]]
    if pivot == 0.0 {
      raise @err.SparseError::SingularPivot(i, pivot)
    }
    z[i] = acc / pivot
  }
  z
}

///|
/// A preconditioner, chosen at run time.
///
/// This is an enum rather than a trait so that a caller can select one from
/// configuration and so that every built-in variant is visible in the API.
pub(all) enum Precond {
  /// No preconditioning: `M^-1 r = r`.
  Identity
  /// Diagonal scaling.
  Jacobi(JacobiPrecond)
  /// Incomplete LU with zero fill.
  Ilu0(Ilu0Precond)
} derive(Debug)

///|
/// Apply the preconditioner: `z = M^-1 r`.
pub fn Precond::apply(
  self : Precond,
  r : Array[Double],
) -> Array[Double] raise @err.SparseError {
  match self {
    Identity => r.copy()
    Jacobi(p) => p.apply(r)
    Ilu0(p) => p.apply(r)
  }
}

///|
/// Human-readable name, for logging and for error messages.
pub fn Precond::name(self : Precond) -> String {
  match self {
    Identity => "identity"
    Jacobi(_) => "jacobi"
    Ilu0(_) => "ilu0"
  }
}

///|
/// Pick a preconditioner for a matrix, falling back to none when the requested
/// one is not defined for it.
///
/// ILU(0) needs a full diagonal; a matrix without one is not an error worth
/// failing a solve over, so the caller gets `Identity` instead and can decide.
pub fn Precond::for_matrix(
  kind : String,
  a : @csr.Csr[Double],
) -> Precond raise @err.SparseError {
  if kind == "identity" {
    return Identity
  }
  if kind == "jacobi" {
    return Jacobi(JacobiPrecond::new(a))
  }
  if kind == "ilu0" {
    match Ilu0Precond::new(a) {
      p => Ilu0(p)
    }
  } else {
    raise @err.SparseError::UnsupportedMatrix(
      "Precond::for_matrix: unknown preconditioner '" +
      kind +
      "', expected identity, jacobi or ilu0",
    )
  }
}
