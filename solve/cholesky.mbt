// Sparse Cholesky factorisation.
//
// The iterative solvers in `krylov.mbt` need one product per iteration and can
// stagnate on ill-conditioned systems. A direct factorisation has neither
// problem: it is exact up to rounding, and once `A = L L^T` is available every
// subsequent right-hand side costs one pair of triangular solves.
//
// The price is fill-in. Eliminating a sparse symmetric matrix generally
// produces a factor with more non-zeros than the original, and the amount
// depends on the order the unknowns are eliminated in — which is what the
// `reorder` package addresses.
//
// The algorithm is the up-looking one: for each column k, compute the pattern
// of row k of L from the elimination tree, then solve the small dense system
// implied by the already-computed leading block. Only the *upper* triangular
// part of `A` is read; the lower part is ignored, so a caller with a full
// symmetric matrix can pass it directly.

///|
/// A sparse Cholesky factor `L` such that `A = L L^T`, stored by columns.
///
/// The diagonal of each column is stored first, at `col_ptr[j]`, followed by
/// the off-diagonal entries in increasing row order.
pub struct CholeskyFactor {
  n : Int
  col_ptr : Array[Int]
  counts : Array[Int]
  row_idx : Array[Int]
  vals : Array[Double]
} derive(Debug)

///|
/// Number of rows and columns.
pub fn CholeskyFactor::n(self : CholeskyFactor) -> Int {
  self.n
}

///|
/// Number of stored entries in `L`, which is at least `n` and generally more
/// than the number of entries in `A` because of fill-in.
pub fn CholeskyFactor::nnz(self : CholeskyFactor) -> Int {
  self.vals.length()
}

///|
/// Read-only view of the column pointers.
pub fn CholeskyFactor::col_ptr(self : CholeskyFactor) -> Array[Int] {
  self.col_ptr
}

///|
/// Read-only view of the row indices.
pub fn CholeskyFactor::row_idx(self : CholeskyFactor) -> Array[Int] {
  self.row_idx
}

///|
/// Read-only view of the factor values.
pub fn CholeskyFactor::vals(self : CholeskyFactor) -> Array[Double] {
  self.vals
}

///|
/// Number of stored entries in column `j`.
pub fn CholeskyFactor::col_nnz(
  self : CholeskyFactor,
  j : Int,
) -> Int raise @err.SparseError {
  if j < 0 || j >= self.n {
    raise @err.SparseError::IndexOutOfBounds(0, j, self.n, self.n)
  }
  self.counts[j]
}

///|
/// The diagonal entry `L(j,j)`.
pub fn CholeskyFactor::diag(
  self : CholeskyFactor,
  j : Int,
) -> Double raise @err.SparseError {
  if j < 0 || j >= self.n {
    raise @err.SparseError::IndexOutOfBounds(j, j, self.n, self.n)
  }
  self.vals[self.col_ptr[j]]
}

///|
/// `log(det(A))`, from `det(A) = prod_j L(j,j)^2`.
///
/// Returned as a logarithm because the determinant of a large sparse matrix
/// overflows long before the matrix stops being interesting.
pub fn CholeskyFactor::log_determinant(self : CholeskyFactor) -> Double {
  let mut acc = 0.0
  for j = 0; j < self.n; j = j + 1 {
    acc = acc + @math.ln(self.vals[self.col_ptr[j]])
  }
  acc * 2.0
}

///|
/// The elimination tree of `A`, given in CSC form.
///
/// `parent[i]` is the row index of the first off-diagonal entry of column `i`
/// of `L`, or -1 for a root. Only entries above the diagonal in each column are
/// examined, so the tree reflects the upper triangular part.
fn elimination_tree(
  n : Int,
  col_ptr : Array[Int],
  row_idx : Array[Int],
) -> Array[Int] {
  let parent : Array[Int] = Array::make(n, -1)
  let ancestor : Array[Int] = Array::make(n, -1)
  for k = 0; k < n; k = k + 1 {
    for p = col_ptr[k]; p < col_ptr[k + 1]; p = p + 1 {
      let mut i = row_idx[p]
      // Walk up the partially built tree, path-compressing as we go.
      while i != -1 && i < k {
        let next = ancestor[i]
        ancestor[i] = k
        if next == -1 {
          parent[i] = k
        }
        i = next
      }
    }
  }
  parent
}

///|
/// The pattern of row `k` of `L`, written into `s` in ascending order ending at
/// `n - 1`. Returns the index of the first used slot.
///
/// This is the "reach" of `k` in the elimination tree: the columns `j < k` for
/// which `L(k,j)` is structurally non-zero. `mark` carries a per-column stamp so
/// that the traversal can tell whether a node was already visited during *this*
/// call without clearing the array in between.
fn ereach(
  n : Int,
  col_ptr : Array[Int],
  row_idx : Array[Int],
  k : Int,
  parent : Array[Int],
  s : Array[Int],
  mark : Array[Int],
) -> Int {
  let mut top = n
  mark[k] = k
  for p = col_ptr[k]; p < col_ptr[k + 1]; p = p + 1 {
    let mut i = row_idx[p]
    let mut len = 0
    while i != -1 && i < k && mark[i] != k {
      s[len] = i
      len = len + 1
      mark[i] = k
      i = parent[i]
    }
    // The collected path is in reverse order; move it to the top of `s` so the
    // final pattern reads in ascending column order.
    while len > 0 {
      len = len - 1
      top = top - 1
      s[top] = s[len]
    }
  }
  top
}

///|
/// Factorise a symmetric positive definite matrix.
///
/// Only the upper triangular part of `a` is used. The factorisation raises
/// `NotPositiveDefinite` when a pivot is not strictly positive, which is the
/// point at which the matrix stops being positive definite — the check is
/// inherent to the algorithm rather than an extra pass.
pub fn CholeskyFactor::new(
  a : @csr.Csr[Double],
) -> CholeskyFactor raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  let n = a.nrows()
  if n == 0 {
    return { n: 0, col_ptr: [0], counts: [], row_idx: [], vals: [], }
  }
  // Column-oriented access is what the algorithm needs, so convert once.
  let csc = @csc.Csc::from_csr(a)
  let acp = csc.col_ptr()
  let ari = csc.row_idx()
  let ava = csc.vals()
  let parent = elimination_tree(n, acp, ari)
  let s : Array[Int] = Array::make(n, 0)
  let mark : Array[Int] = Array::make(n, -1)
  // Symbolic pass: count the entries each column of L will hold, diagonal
  // included. Doing this up front lets the numeric pass write into a flat array
  // without ever growing it.
  let counts : Array[Int] = Array::make(n, 1)
  for k = 0; k < n; k = k + 1 {
    let top = ereach(n, acp, ari, k, parent, s, mark)
    for sp = top; sp < n; sp = sp + 1 {
      counts[s[sp]] = counts[s[sp]] + 1
    }
  }
  let col_ptr : Array[Int] = Array::make(n + 1, 0)
  for j = 0; j < n; j = j + 1 {
    col_ptr[j + 1] = col_ptr[j] + counts[j]
  }
  let total = col_ptr[n]
  let row_idx : Array[Int] = Array::make(total, 0)
  let vals : Array[Double] = Array::make(total, 0.0)
  // `fill[j]` is where the next off-diagonal entry of column j goes.
  let fill : Array[Int] = Array::make(n, 0)
  for j = 0; j < n; j = j + 1 {
    fill[j] = col_ptr[j] + 1
  }
  // Dense workspace for the current column, cleared as it is consumed.
  let x : Array[Double] = Array::make(n, 0.0)
  for k = 0; k < n; k = k + 1 {
    let top = ereach(n, acp, ari, k, parent, s, mark)
    // Scatter the upper triangular part of column k of A into the workspace.
    for p = acp[k]; p < acp[k + 1]; p = p + 1 {
      if ari[p] <= k {
        x[ari[p]] = ava[p]
      }
    }
    let mut d = x[k]
    x[k] = 0.0
    for sp = top; sp < n; sp = sp + 1 {
      let i = s[sp]
      let lii = vals[col_ptr[i]]
      if lii == 0.0 {
        raise @err.SparseError::SingularPivot(i, lii)
      }
      let lki = x[i] / lii
      x[i] = 0.0
      // Update the workspace with the already-computed part of row i of L.
      for p = col_ptr[i] + 1; p < fill[i]; p = p + 1 {
        x[row_idx[p]] = x[row_idx[p]] - vals[p] * lki
      }
      d = d - lki * lki
      row_idx[fill[i]] = k
      vals[fill[i]] = lki
      fill[i] = fill[i] + 1
    }
    if !(d > 0.0) {
      raise @err.SparseError::NotPositiveDefinite(k, d)
    }
    row_idx[col_ptr[k]] = k
    vals[col_ptr[k]] = Double::sqrt(d)
  }
  { n, col_ptr, counts, row_idx, vals, }
}

///|
/// Factorise, verifying first that the matrix really is symmetric.
///
/// The symmetry check is a pass over the stored entries with a binary search
/// each, which is negligible next to the factorisation itself and removes the
/// silent failure mode `new` has. Use this unless the matrix is known
/// symmetric or only one triangle is stored.
pub fn CholeskyFactor::new_checked(
  a : @csr.Csr[Double],
) -> CholeskyFactor raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  if !a.is_symmetric() {
    raise @err.SparseError::UnsupportedMatrix(
      "CholeskyFactor::new_checked: the matrix is not symmetric, so a Cholesky factor of it does not exist",
    )
  }
  CholeskyFactor::new(a)
}

///|
/// Solve `A x = b` using the factorisation, by a forward solve with `L` and a
/// backward solve with `L^T`.
pub fn CholeskyFactor::solve(
  self : CholeskyFactor,
  b : Array[Double],
) -> Array[Double] raise @err.SparseError {
  let n = self.n
  if b.length() != n {
    raise @err.SparseError::DimensionMismatch(
      "CholeskyFactor::solve: b has length " +
      b.length().to_string() +
      ", expected " +
      n.to_string(),
    )
  }
  let cp = self.col_ptr
  let counts = self.counts
  let ri = self.row_idx
  let va = self.vals
  // Forward solve L y = b. Column-oriented, so each y[j] is finalised before it
  // is used to update the entries below it.
  let y = b.copy()
  for j = 0; j < n; j = j + 1 {
    let d = va[cp[j]]
    if d == 0.0 {
      raise @err.SparseError::SingularPivot(j, d)
    }
    y[j] = y[j] / d
    let yj = y[j]
    for p = cp[j] + 1; p < cp[j] + counts[j]; p = p + 1 {
      y[ri[p]] = y[ri[p]] - va[p] * yj
    }
  }
  // Backward solve L^T x = y. Here each row of L^T is a column of L, so the
  // entries of column j are read as the row of L^T, and the order is reversed.
  for jj = 0; jj < n; jj = jj + 1 {
    let j = n - 1 - jj
    let mut acc = y[j]
    for p = cp[j] + 1; p < cp[j] + counts[j]; p = p + 1 {
      acc = acc - va[p] * y[ri[p]]
    }
    let d = va[cp[j]]
    if d == 0.0 {
      raise @err.SparseError::SingularPivot(j, d)
    }
    y[j] = acc / d
  }
  y
}

///|
/// Reconstruct `L L^T` as a dense row-major array.
///
/// The factorisation is only useful if it multiplies back to the matrix it came
/// from, and this is how that is checked. It allocates `n^2` entries, so it is a
/// verification aid, not a working routine.
pub fn CholeskyFactor::reconstruct(self : CholeskyFactor) -> Array[Double] {
  let n = self.n
  let dense : Array[Double] = Array::make(n * n, 0.0)
  // Accumulate sum_j L(i,j) L(k,j) over the columns, which is exactly (L L^T).
  for j = 0; j < n; j = j + 1 {
    for p = self.col_ptr[j]; p < self.col_ptr[j] + self.counts[j]; p = p + 1 {
      let i = self.row_idx[p]
      let lj = self.vals[p]
      for q = self.col_ptr[j]; q < self.col_ptr[j] + self.counts[j]; q = q + 1 {
        let k = self.row_idx[q]
        dense[i * n + k] = dense[i * n + k] + lj * self.vals[q]
      }
    }
  }
  dense
}

///|
/// Factorise and solve in one call, for a single right-hand side.
pub fn solve_cholesky(
  a : @csr.Csr[Double],
  b : Array[Double],
) -> Array[Double] raise @err.SparseError {
  CholeskyFactor::new(a).solve(b)
}
