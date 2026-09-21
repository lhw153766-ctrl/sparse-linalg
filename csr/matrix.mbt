// Compressed Sparse Row storage.
//
// CSR keeps three parallel arrays:
//
//   row_ptr : length nrows + 1, row_ptr[i] is the first slot of row i
//   col_idx : length nnz, the column of each stored entry
//   vals    : length nnz, the value of each stored entry
//
// The column indices inside one row are kept sorted, which makes `get` a
// binary search and keeps row unions cheap. `Csr::new` checks every invariant
// rather than trusting the caller.

///|
/// A matrix in compressed sparse row form.
pub struct Csr[F] {
  nrows : Int
  ncols : Int
  row_ptr : Array[Int]
  col_idx : Array[Int]
  vals : Array[F]
}

///|
/// Build a CSR matrix, checking all structural invariants.
///
/// Rejects `row_ptr` of the wrong length, non-monotone `row_ptr`, mismatched
/// `col_idx`/`vals` lengths, out-of-range column indices, and unsorted or
/// duplicated columns within a row. Use `Csr::new_unchecked` on hot paths
/// where the arrays were just produced by a builder that already upholds them.
pub fn[F] Csr::new(
  nrows : Int,
  ncols : Int,
  row_ptr : Array[Int],
  col_idx : Array[Int],
  vals : Array[F],
) -> Csr[F] raise @err.SparseError {
  if nrows < 0 || ncols < 0 {
    raise @err.SparseError::DimensionMismatch(
      "Csr::new: negative shape " + nrows.to_string() + "x" + ncols.to_string(),
    )
  }
  if row_ptr.length() != nrows + 1 {
    raise @err.SparseError::DimensionMismatch(
      "Csr::new: row_ptr has length " +
      row_ptr.length().to_string() +
      ", expected " +
      (nrows + 1).to_string(),
    )
  }
  if col_idx.length() != vals.length() {
    raise @err.SparseError::DimensionMismatch(
      "Csr::new: col_idx has length " +
      col_idx.length().to_string() +
      " but vals has length " +
      vals.length().to_string(),
    )
  }
  if row_ptr[0] != 0 {
    raise @err.SparseError::DimensionMismatch("Csr::new: row_ptr[0] must be 0")
  }
  if row_ptr[nrows] != col_idx.length() {
    raise @err.SparseError::DimensionMismatch(
      "Csr::new: row_ptr[nrows] is " +
      row_ptr[nrows].to_string() +
      " but nnz is " +
      col_idx.length().to_string(),
    )
  }
  for i = 0; i < nrows; i = i + 1 {
    if row_ptr[i] > row_ptr[i + 1] {
      raise @err.SparseError::DimensionMismatch(
        "Csr::new: row_ptr decreases at row " + i.to_string(),
      )
    }
  }
  for i = 0; i < nrows; i = i + 1 {
    for k = row_ptr[i]; k < row_ptr[i + 1]; k = k + 1 {
      let c = col_idx[k]
      if c < 0 || c >= ncols {
        raise @err.SparseError::IndexOutOfBounds(i, c, nrows, ncols)
      }
      if k > row_ptr[i] && col_idx[k - 1] >= c {
        raise @err.SparseError::DimensionMismatch(
          "Csr::new: columns not strictly increasing in row " + i.to_string(),
        )
      }
    }
  }
  { nrows, ncols, row_ptr, col_idx, vals, }
}

///|
/// Build a CSR matrix from arrays that are already known to be valid.
///
/// Skips validation entirely. Intended for builders inside this module and for
/// callers who have just sorted and deduplicated their own triplets.
pub fn[F] Csr::new_unchecked(
  nrows : Int,
  ncols : Int,
  row_ptr : Array[Int],
  col_idx : Array[Int],
  vals : Array[F],
) -> Csr[F] {
  { nrows, ncols, row_ptr, col_idx, vals, }
}

///|
/// Number of rows.
pub fn[F] Csr::nrows(self : Csr[F]) -> Int {
  self.nrows
}

///|
/// Number of columns.
pub fn[F] Csr::ncols(self : Csr[F]) -> Int {
  self.ncols
}

///|
/// Number of stored entries.
pub fn[F] Csr::nnz(self : Csr[F]) -> Int {
  self.vals.length()
}

///|
/// The declared shape.
pub fn[F] Csr::shape(self : Csr[F]) -> @err.Shape {
  { rows: self.nrows, cols: self.ncols, }
}

///|
/// Read-only view of the row pointer array.
pub fn[F] Csr::row_ptr(self : Csr[F]) -> Array[Int] {
  self.row_ptr
}

///|
/// Read-only view of the column index array.
pub fn[F] Csr::col_idx(self : Csr[F]) -> Array[Int] {
  self.col_idx
}

///|
/// Read-only view of the value array.
pub fn[F] Csr::vals(self : Csr[F]) -> Array[F] {
  self.vals
}

///|
/// Number of stored entries in row `i`.
pub fn[F] Csr::row_nnz(self : Csr[F], i : Int) -> Int raise @err.SparseError {
  if i < 0 || i >= self.nrows {
    raise @err.SparseError::IndexOutOfBounds(i, 0, self.nrows, self.ncols)
  }
  self.row_ptr[i + 1] - self.row_ptr[i]
}

///|
/// The half-open slot range `[start, end)` holding row `i`.
pub fn[F] Csr::row_range(
  self : Csr[F],
  i : Int,
) -> (Int, Int) raise @err.SparseError {
  if i < 0 || i >= self.nrows {
    raise @err.SparseError::IndexOutOfBounds(i, 0, self.nrows, self.ncols)
  }
  (self.row_ptr[i], self.row_ptr[i + 1])
}

///|
/// Position of column `j` inside row `i`, or `None` when the entry is absent.
///
/// Binary search, so this is `O(log row_nnz)`.
pub fn[F] Csr::find(
  self : Csr[F],
  i : Int,
  j : Int,
) -> Int? raise @err.SparseError {
  if i < 0 || i >= self.nrows || j < 0 || j >= self.ncols {
    raise @err.SparseError::IndexOutOfBounds(i, j, self.nrows, self.ncols)
  }
  let mut lo = self.row_ptr[i]
  let mut hi = self.row_ptr[i + 1]
  while lo < hi {
    let mid = lo + (hi - lo) / 2
    let c = self.col_idx[mid]
    if c == j {
      return Some(mid)
    } else if c < j {
      lo = mid + 1
    } else {
      hi = mid
    }
  }
  None
}

///|
/// Read entry `(i, j)`, or `None` when it is not stored.
pub fn[F] Csr::get(
  self : Csr[F],
  i : Int,
  j : Int,
) -> F? raise @err.SparseError {
  match self.find(i, j) {
    Some(k) => Some(self.vals[k])
    None => None
  }
}

///|
/// The stored value at `(i, j)`, or `fallback` when the entry is absent.
///
/// This is the dense-style read: a caller that wants "zero when missing"
/// passes `Default::default()`.
pub fn[F] Csr::get_or(
  self : Csr[F],
  i : Int,
  j : Int,
  fallback : F,
) -> F raise @err.SparseError {
  match self.find(i, j) {
    Some(k) => self.vals[k]
    None => fallback
  }
}

///|
/// True when `(i, j)` is explicitly stored.
pub fn[F] Csr::contains(
  self : Csr[F],
  i : Int,
  j : Int,
) -> Bool raise @err.SparseError {
  match self.find(i, j) {
    Some(_) => true
    None => false
  }
}

///|
/// The main sparse kernel: `y = A * x`.
///
/// Cost is `O(nnz)`, independent of how many zeros the dense form would hold.
pub fn[F : Add + Mul + Default] Csr::spmv(
  self : Csr[F],
  x : Array[F],
) -> Array[F] raise @err.SparseError {
  if x.length() != self.ncols {
    raise @err.SparseError::DimensionMismatch(
      "Csr::spmv: x has length " +
      x.length().to_string() +
      " but the matrix has " +
      self.ncols.to_string() +
      " columns",
    )
  }
  let y : Array[F] = Array::make(self.nrows, Default::default())
  for i = 0; i < self.nrows; i = i + 1 {
    let mut acc : F = Default::default()
    for k = self.row_ptr[i]; k < self.row_ptr[i + 1]; k = k + 1 {
      acc = acc + self.vals[k] * x[self.col_idx[k]]
    }
    y[i] = acc
  }
  y
}

///|
/// `y = alpha * A * x + beta * y`, the BLAS-style kernel that iterative
/// solvers and preconditioners want. `y` is updated in place.
pub fn[F : Add + Mul + Default] Csr::spmv_add(
  self : Csr[F],
  alpha : F,
  x : Array[F],
  beta : F,
  y : Array[F],
) -> Unit raise @err.SparseError {
  if x.length() != self.ncols {
    raise @err.SparseError::DimensionMismatch(
      "Csr::spmv_add: x has length " +
      x.length().to_string() +
      ", expected " +
      self.ncols.to_string(),
    )
  }
  if y.length() != self.nrows {
    raise @err.SparseError::DimensionMismatch(
      "Csr::spmv_add: y has length " +
      y.length().to_string() +
      ", expected " +
      self.nrows.to_string(),
    )
  }
  for i = 0; i < self.nrows; i = i + 1 {
    let mut acc : F = Default::default()
    for k = self.row_ptr[i]; k < self.row_ptr[i + 1]; k = k + 1 {
      acc = acc + self.vals[k] * x[self.col_idx[k]]
    }
    y[i] = alpha * acc + beta * y[i]
  }
}

///|
/// Sum of every stored value. Uses only `Add`, so exact scalar types such as
/// `Int64` keep their exactness.
pub fn[F : Add + Default] Csr::sum_vals(self : Csr[F]) -> F {
  let mut acc : F = Default::default()
  for k = 0; k < self.vals.length(); k = k + 1 {
    acc = acc + self.vals[k]
  }
  acc
}

///|
/// The main diagonal, including the implicit zeros.
///
/// Length is `min(nrows, ncols)`.
pub fn[F : Default] Csr::diagonal(
  self : Csr[F],
) -> Array[F] raise @err.SparseError {
  let n = if self.nrows < self.ncols { self.nrows } else { self.ncols }
  let d : Array[F] = Array::make(n, Default::default())
  for i = 0; i < n; i = i + 1 {
    match self.find(i, i) {
      Some(k) => d[i] = self.vals[k]
      None => ()
    }
  }
  d
}

///|
/// True when the stored pattern is symmetric and every mirrored pair holds the
/// same value. Both `(i, j)` and `(j, i)` must be stored.
///
/// Solvers that require symmetry call this instead of assuming it.
pub fn[F : Eq] Csr::is_symmetric(self : Csr[F]) -> Bool raise @err.SparseError {
  if self.nrows != self.ncols {
    return false
  }
  for i = 0; i < self.nrows; i = i + 1 {
    for k = self.row_ptr[i]; k < self.row_ptr[i + 1]; k = k + 1 {
      let j = self.col_idx[k]
      match self.find(j, i) {
        Some(k2) => if self.vals[k] != self.vals[k2] { return false }
        None => return false
      }
    }
  }
  true
}

///|
/// Expand to a dense row-major array of length `nrows * ncols`.
///
/// Intended for tests, small problems, and as the bridge to dense reference
/// implementations. It is `O(nrows * ncols)` in time and memory, so it is the
/// wrong tool once the matrix is large enough to need sparsity.
pub fn[F : Default] Csr::to_dense(self : Csr[F]) -> Array[F] {
  let dense : Array[F] = Array::make(
    self.nrows * self.ncols,
    Default::default(),
  )
  for i = 0; i < self.nrows; i = i + 1 {
    for k = self.row_ptr[i]; k < self.row_ptr[i + 1]; k = k + 1 {
      dense[i * self.ncols + self.col_idx[k]] = self.vals[k]
    }
  }
  dense
}

///|
/// Multiply every stored value by `alpha`.
pub fn[F : Mul] Csr::scale(self : Csr[F], alpha : F) -> Csr[F] {
  let out : Array[F] = Array(capacity=0)
  for k = 0; k < self.vals.length(); k = k + 1 {
    out.push(alpha * self.vals[k])
  }
  {
    nrows: self.nrows,
    ncols: self.ncols,
    row_ptr: self.row_ptr.copy(),
    col_idx: self.col_idx.copy(),
    vals: out,
  }
}

///|
/// Number of structurally non-zero entries per row.
pub fn[F] Csr::row_nnz_all(self : Csr[F]) -> Array[Int] {
  let out : Array[Int] = Array::make(self.nrows, 0)
  for i = 0; i < self.nrows; i = i + 1 {
    out[i] = self.row_ptr[i + 1] - self.row_ptr[i]
  }
  out
}

///|
/// Largest number of stored entries in any row.
pub fn[F] Csr::max_row_nnz(self : Csr[F]) -> Int {
  let mut m = 0
  for i = 0; i < self.nrows; i = i + 1 {
    let n = self.row_ptr[i + 1] - self.row_ptr[i]
    if n > m {
      m = n
    }
  }
  m
}
