// Compressed Sparse Column storage.
//
// CSC mirrors CSR with the roles of rows and columns swapped:
//
//   col_ptr : length ncols + 1, col_ptr[j] is the first slot of column j
//   row_idx : length nnz, the row of each stored entry
//   vals    : length nnz, the value of each stored entry
//
// It exists for two reasons. Column-oriented reads ("give me the entries of
// column j") are a scan in CSR and a slice in CSC, and the product `A^T x` is a
// plain SpMV in CSC while it needs a transpose in CSR. BiCGSTAB and GMRES use
// `A^T x` on non-symmetric systems, so having both layouts avoids rebuilding a
// transposed matrix on every iteration.

///|
/// A matrix in compressed sparse column form.
pub struct Csc[F] {
  nrows : Int
  ncols : Int
  col_ptr : Array[Int]
  row_idx : Array[Int]
  vals : Array[F]
}

///|
/// Build a CSC matrix, checking all structural invariants.
///
/// Rejects a `col_ptr` of the wrong length, a `col_ptr` that does not start at
/// 0 or end at `nnz`, a decreasing `col_ptr`, mismatched `row_idx`/`vals`
/// lengths, out-of-range rows, and unsorted or duplicated rows within a column.
pub fn[F] Csc::new(
  nrows : Int,
  ncols : Int,
  col_ptr : Array[Int],
  row_idx : Array[Int],
  vals : Array[F],
) -> Csc[F] raise @err.SparseError {
  if nrows < 0 || ncols < 0 {
    raise @err.SparseError::DimensionMismatch(
      "Csc::new: negative shape " + nrows.to_string() + "x" + ncols.to_string(),
    )
  }
  if col_ptr.length() != ncols + 1 {
    raise @err.SparseError::DimensionMismatch(
      "Csc::new: col_ptr has length " +
      col_ptr.length().to_string() +
      ", expected " +
      (ncols + 1).to_string(),
    )
  }
  if row_idx.length() != vals.length() {
    raise @err.SparseError::DimensionMismatch(
      "Csc::new: row_idx has length " +
      row_idx.length().to_string() +
      " but vals has length " +
      vals.length().to_string(),
    )
  }
  if col_ptr[0] != 0 {
    raise @err.SparseError::DimensionMismatch("Csc::new: col_ptr[0] must be 0")
  }
  if col_ptr[ncols] != row_idx.length() {
    raise @err.SparseError::DimensionMismatch(
      "Csc::new: col_ptr[ncols] is " +
      col_ptr[ncols].to_string() +
      " but nnz is " +
      row_idx.length().to_string(),
    )
  }
  for j = 0; j < ncols; j = j + 1 {
    if col_ptr[j] > col_ptr[j + 1] {
      raise @err.SparseError::DimensionMismatch(
        "Csc::new: col_ptr decreases at column " + j.to_string(),
      )
    }
  }
  for j = 0; j < ncols; j = j + 1 {
    for k = col_ptr[j]; k < col_ptr[j + 1]; k = k + 1 {
      let r = row_idx[k]
      if r < 0 || r >= nrows {
        raise @err.SparseError::IndexOutOfBounds(r, j, nrows, ncols)
      }
      if k > col_ptr[j] && row_idx[k - 1] >= r {
        raise @err.SparseError::DimensionMismatch(
          "Csc::new: rows not strictly increasing in column " + j.to_string(),
        )
      }
    }
  }
  { nrows, ncols, col_ptr, row_idx, vals, }
}

///|
/// Build a CSC matrix from arrays already known to be valid.
pub fn[F] Csc::new_unchecked(
  nrows : Int,
  ncols : Int,
  col_ptr : Array[Int],
  row_idx : Array[Int],
  vals : Array[F],
) -> Csc[F] {
  { nrows, ncols, col_ptr, row_idx, vals, }
}

///|
/// Number of rows.
pub fn[F] Csc::nrows(self : Csc[F]) -> Int {
  self.nrows
}

///|
/// Number of columns.
pub fn[F] Csc::ncols(self : Csc[F]) -> Int {
  self.ncols
}

///|
/// Number of stored entries.
pub fn[F] Csc::nnz(self : Csc[F]) -> Int {
  self.vals.length()
}

///|
/// The declared shape.
pub fn[F] Csc::shape(self : Csc[F]) -> @err.Shape {
  { rows: self.nrows, cols: self.ncols, }
}

///|
/// Read-only view of the column pointer array.
pub fn[F] Csc::col_ptr(self : Csc[F]) -> Array[Int] {
  self.col_ptr
}

///|
/// Read-only view of the row index array.
pub fn[F] Csc::row_idx(self : Csc[F]) -> Array[Int] {
  self.row_idx
}

///|
/// Read-only view of the value array.
pub fn[F] Csc::vals(self : Csc[F]) -> Array[F] {
  self.vals
}

///|
/// Number of stored entries in column `j`.
pub fn[F] Csc::col_nnz(self : Csc[F], j : Int) -> Int raise @err.SparseError {
  if j < 0 || j >= self.ncols {
    raise @err.SparseError::IndexOutOfBounds(0, j, self.nrows, self.ncols)
  }
  self.col_ptr[j + 1] - self.col_ptr[j]
}

///|
/// The half-open slot range `[start, end)` holding column `j`.
pub fn[F] Csc::col_range(
  self : Csc[F],
  j : Int,
) -> (Int, Int) raise @err.SparseError {
  if j < 0 || j >= self.ncols {
    raise @err.SparseError::IndexOutOfBounds(0, j, self.nrows, self.ncols)
  }
  (self.col_ptr[j], self.col_ptr[j + 1])
}

///|
/// Position of row `i` inside column `j`, or `None` when the entry is absent.
pub fn[F] Csc::find(
  self : Csc[F],
  i : Int,
  j : Int,
) -> Int? raise @err.SparseError {
  if i < 0 || i >= self.nrows || j < 0 || j >= self.ncols {
    raise @err.SparseError::IndexOutOfBounds(i, j, self.nrows, self.ncols)
  }
  let mut lo = self.col_ptr[j]
  let mut hi = self.col_ptr[j + 1]
  while lo < hi {
    let mid = lo + (hi - lo) / 2
    let r = self.row_idx[mid]
    if r == i {
      return Some(mid)
    } else if r < i {
      lo = mid + 1
    } else {
      hi = mid
    }
  }
  None
}

///|
/// Read entry `(i, j)`, or `None` when it is not stored.
pub fn[F] Csc::get(
  self : Csc[F],
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
pub fn[F] Csc::get_or(
  self : Csc[F],
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
pub fn[F] Csc::contains(
  self : Csc[F],
  i : Int,
  j : Int,
) -> Bool raise @err.SparseError {
  match self.find(i, j) {
    Some(_) => true
    None => false
  }
}

///|
/// `y = A * x`.
///
/// Column-oriented: every stored entry of column `j` contributes `vals[k]*x[j]`
/// to `y[row_idx[k]]`. Scatter rather than gather, which is why it is a
/// separate implementation from the CSR one.
pub fn[F : Add + Mul + Default] Csc::spmv(
  self : Csc[F],
  x : Array[F],
) -> Array[F] raise @err.SparseError {
  if x.length() != self.ncols {
    raise @err.SparseError::DimensionMismatch(
      "Csc::spmv: x has length " +
      x.length().to_string() +
      " but the matrix has " +
      self.ncols.to_string() +
      " columns",
    )
  }
  let y : Array[F] = Array::make(self.nrows, Default::default())
  for j = 0; j < self.ncols; j = j + 1 {
    let xj = x[j]
    for k = self.col_ptr[j]; k < self.col_ptr[j + 1]; k = k + 1 {
      let r = self.row_idx[k]
      y[r] = y[r] + self.vals[k] * xj
    }
  }
  y
}

///|
/// `y = A^T * x`.
///
/// In CSC this is the natural direction: column `j` of `A` is row `j` of `A^T`,
/// so the result entry `y[j]` is the inner product of a contiguous slice of
/// `vals` with `x`.
pub fn[F : Add + Mul + Default] Csc::spmv_transpose(
  self : Csc[F],
  x : Array[F],
) -> Array[F] raise @err.SparseError {
  if x.length() != self.nrows {
    raise @err.SparseError::DimensionMismatch(
      "Csc::spmv_transpose: x has length " +
      x.length().to_string() +
      " but the matrix has " +
      self.nrows.to_string() +
      " rows",
    )
  }
  let y : Array[F] = Array::make(self.ncols, Default::default())
  for j = 0; j < self.ncols; j = j + 1 {
    let mut acc : F = Default::default()
    for k = self.col_ptr[j]; k < self.col_ptr[j + 1]; k = k + 1 {
      acc = acc + self.vals[k] * x[self.row_idx[k]]
    }
    y[j] = acc
  }
  y
}

///|
/// Sum of every stored value.
pub fn[F : Add + Default] Csc::sum_vals(self : Csc[F]) -> F {
  let mut acc : F = Default::default()
  for k = 0; k < self.vals.length(); k = k + 1 {
    acc = acc + self.vals[k]
  }
  acc
}

///|
/// The main diagonal, including implicit zeros. Length is `min(nrows, ncols)`.
pub fn[F : Default] Csc::diagonal(
  self : Csc[F],
) -> Array[F] raise @err.SparseError {
  let n = if self.nrows < self.ncols { self.nrows } else { self.ncols }
  let d : Array[F] = Array::make(n, Default::default())
  for j = 0; j < n; j = j + 1 {
    match self.find(j, j) {
      Some(k) => d[j] = self.vals[k]
      None => ()
    }
  }
  d
}

///|
/// Expand to a dense row-major array of length `nrows * ncols`.
pub fn[F : Default] Csc::to_dense(self : Csc[F]) -> Array[F] {
  let dense : Array[F] = Array::make(
    self.nrows * self.ncols,
    Default::default(),
  )
  for j = 0; j < self.ncols; j = j + 1 {
    for k = self.col_ptr[j]; k < self.col_ptr[j + 1]; k = k + 1 {
      dense[self.row_idx[k] * self.ncols + j] = self.vals[k]
    }
  }
  dense
}

///|
/// Multiply every stored value by `alpha`.
pub fn[F : Mul] Csc::scale(self : Csc[F], alpha : F) -> Csc[F] {
  let out : Array[F] = Array(capacity=self.vals.length())
  for k = 0; k < self.vals.length(); k = k + 1 {
    out.push(alpha * self.vals[k])
  }
  {
    nrows: self.nrows,
    ncols: self.ncols,
    col_ptr: self.col_ptr.copy(),
    row_idx: self.row_idx.copy(),
    vals: out,
  }
}

///|
/// Largest number of stored entries in any column.
pub fn[F] Csc::max_col_nnz(self : Csc[F]) -> Int {
  let mut m = 0
  for j = 0; j < self.ncols; j = j + 1 {
    let n = self.col_ptr[j + 1] - self.col_ptr[j]
    if n > m {
      m = n
    }
  }
  m
}

///|
/// True when the stored pattern is symmetric and mirrored values agree.
pub fn[F : Eq] Csc::is_symmetric(self : Csc[F]) -> Bool raise @err.SparseError {
  if self.nrows != self.ncols {
    return false
  }
  for j = 0; j < self.ncols; j = j + 1 {
    for k = self.col_ptr[j]; k < self.col_ptr[j + 1]; k = k + 1 {
      let i = self.row_idx[k]
      match self.find(j, i) {
        Some(k2) => if self.vals[k] != self.vals[k2] { return false }
        None => return false
      }
    }
  }
  true
}
