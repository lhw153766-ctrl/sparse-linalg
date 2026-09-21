// Coordinate (triplet) storage.
//
// COO is the format people actually have when a matrix comes from a mesh, an
// edge list, or a file: a flat list of `(row, col, value)` triples. It is a
// poor format to compute in — every lookup is a scan — but it is the right
// format to assemble in and to convert out of.
//
// `Coo::to_csr` is where assembly happens: it counts, sorts, and *sums
// duplicate coordinates*, so a caller can emit the same coordinate more than
// once and get the mathematically correct matrix.

///|
/// One `(row, col, value)` triple.
pub(all) struct Triplet[F] {
  row : Int
  col : Int
  val : F
} derive(Debug, Eq)

///|
/// A matrix held as a list of coordinate triples.
///
/// The arrays are kept in insertion order. Duplicate coordinates are allowed
/// and are summed when converting to CSR or CSC.
pub struct Coo[F] {
  nrows : Int
  ncols : Int
  rows : Array[Int]
  cols : Array[Int]
  vals : Array[F]
}

///|
/// An empty COO matrix with the given shape.
pub fn[F] Coo::empty(nrows : Int, ncols : Int) -> Coo[F] raise @err.SparseError {
  if nrows < 0 || ncols < 0 {
    raise @err.SparseError::DimensionMismatch(
      "Coo::empty: negative shape " +
      nrows.to_string() +
      "x" +
      ncols.to_string(),
    )
  }
  { nrows, ncols, rows: [], cols: [], vals: [], }
}

///|
/// Build from three parallel arrays, checking lengths and bounds.
pub fn[F] Coo::new(
  nrows : Int,
  ncols : Int,
  rows : Array[Int],
  cols : Array[Int],
  vals : Array[F],
) -> Coo[F] raise @err.SparseError {
  if nrows < 0 || ncols < 0 {
    raise @err.SparseError::DimensionMismatch(
      "Coo::new: negative shape " + nrows.to_string() + "x" + ncols.to_string(),
    )
  }
  if rows.length() != cols.length() || rows.length() != vals.length() {
    raise @err.SparseError::DimensionMismatch(
      "Coo::new: rows/cols/vals lengths differ (" +
      rows.length().to_string() +
      ", " +
      cols.length().to_string() +
      ", " +
      vals.length().to_string() +
      ")",
    )
  }
  for k = 0; k < rows.length(); k = k + 1 {
    if rows[k] < 0 || rows[k] >= nrows || cols[k] < 0 || cols[k] >= ncols {
      raise @err.SparseError::IndexOutOfBounds(rows[k], cols[k], nrows, ncols)
    }
  }
  { nrows, ncols, rows, cols, vals, }
}

///|
/// Build from a list of triplets, checking bounds.
pub fn[F] Coo::from_triplets(
  nrows : Int,
  ncols : Int,
  triplets : Array[Triplet[F]],
) -> Coo[F] raise @err.SparseError {
  let rows : Array[Int] = Array(capacity=triplets.length())
  let cols : Array[Int] = Array(capacity=triplets.length())
  let vals : Array[F] = Array(capacity=triplets.length())
  for t in triplets {
    rows.push(t.row)
    cols.push(t.col)
    vals.push(t.val)
  }
  Coo::new(nrows, ncols, rows, cols, vals)
}

///|
/// Append one entry, checking that it lies inside the shape.
pub fn[F] Coo::push(
  self : Coo[F],
  row : Int,
  col : Int,
  val : F,
) -> Unit raise @err.SparseError {
  if row < 0 || row >= self.nrows || col < 0 || col >= self.ncols {
    raise @err.SparseError::IndexOutOfBounds(row, col, self.nrows, self.ncols)
  }
  self.rows.push(row)
  self.cols.push(col)
  self.vals.push(val)
}

///|
/// Number of rows.
pub fn[F] Coo::nrows(self : Coo[F]) -> Int {
  self.nrows
}

///|
/// Number of columns.
pub fn[F] Coo::ncols(self : Coo[F]) -> Int {
  self.ncols
}

///|
/// Number of stored triples, counting duplicates separately.
pub fn[F] Coo::nnz(self : Coo[F]) -> Int {
  self.vals.length()
}

///|
/// The declared shape.
pub fn[F] Coo::shape(self : Coo[F]) -> @err.Shape {
  { rows: self.nrows, cols: self.ncols, }
}

///|
/// Read-only view of the row indices.
pub fn[F] Coo::rows(self : Coo[F]) -> Array[Int] {
  self.rows
}

///|
/// Read-only view of the column indices.
pub fn[F] Coo::cols(self : Coo[F]) -> Array[Int] {
  self.cols
}

///|
/// Read-only view of the values.
pub fn[F] Coo::vals(self : Coo[F]) -> Array[F] {
  self.vals
}

///|
/// Sum of every stored value, duplicates included.
pub fn[F : Add + Default] Coo::sum_vals(self : Coo[F]) -> F {
  let mut acc : F = Default::default()
  for k = 0; k < self.vals.length(); k = k + 1 {
    acc = acc + self.vals[k]
  }
  acc
}

///|
/// Multiply every stored value by `alpha`.
pub fn[F : Mul] Coo::scale(self : Coo[F], alpha : F) -> Coo[F] {
  let out : Array[F] = Array(capacity=self.vals.length())
  for k = 0; k < self.vals.length(); k = k + 1 {
    out.push(alpha * self.vals[k])
  }
  {
    nrows: self.nrows,
    ncols: self.ncols,
    rows: self.rows.copy(),
    cols: self.cols.copy(),
    vals: out,
  }
}

///|
/// The mirror image: `(row, col)` becomes `(col, row)`.
pub fn[F] Coo::transpose(self : Coo[F]) -> Coo[F] {
  {
    nrows: self.ncols,
    ncols: self.nrows,
    rows: self.cols.copy(),
    cols: self.rows.copy(),
    vals: self.vals.copy(),
  }
}

///|
/// The value at `(i, j)`, summing duplicates, or `None` when nothing is stored.
///
/// Linear scan: `O(nnz)`. Present for completeness and small matrices; use CSR
/// for repeated lookups.
pub fn[F : Add + Default] Coo::get(
  self : Coo[F],
  i : Int,
  j : Int,
) -> F? raise @err.SparseError {
  self.shape().check_index(i, j)
  let mut found = false
  let mut acc : F = Default::default()
  for k = 0; k < self.vals.length(); k = k + 1 {
    if self.rows[k] == i && self.cols[k] == j {
      acc = acc + self.vals[k]
      found = true
    }
  }
  if found {
    Some(acc)
  } else {
    None
  }
}

// Sort key used while assembling a row or column: column index, then value
// order is irrelevant because equal columns are summed anyway.

///|
priv struct Slot[F] {
  idx : Int
  val : F
}

///|
/// Convert to CSR, summing duplicate coordinates and sorting each row.
///
/// The result satisfies every invariant `Csr::new` checks, so it is built with
/// `Csr::new_unchecked`.
pub fn[F : Add] Coo::to_csr(self : Coo[F]) -> @csr.Csr[F] {
  let nnz = self.vals.length()
  // 1. count entries per row
  let row_ptr : Array[Int] = Array::make(self.nrows + 1, 0)
  for k = 0; k < nnz; k = k + 1 {
    row_ptr[self.rows[k] + 1] = row_ptr[self.rows[k] + 1] + 1
  }
  // 2. exclusive prefix sum gives each row's start slot
  for i = 0; i < self.nrows; i = i + 1 {
    row_ptr[i + 1] = row_ptr[i + 1] + row_ptr[i]
  }
  // 3. bucket the entries by row
  let buckets : Array[Array[Slot[F]]] = Array::make(self.nrows, [])
  for i = 0; i < self.nrows; i = i + 1 {
    buckets[i] = Array(capacity=row_ptr[i + 1] - row_ptr[i])
  }
  for k = 0; k < nnz; k = k + 1 {
    buckets[self.rows[k]].push({ idx: self.cols[k], val: self.vals[k], })
  }
  // 4. sort each row by column and merge equal columns
  let col_idx : Array[Int] = Array(capacity=nnz)
  let vals : Array[F] = Array(capacity=nnz)
  let merged_ptr : Array[Int] = Array::make(self.nrows + 1, 0)
  for i = 0; i < self.nrows; i = i + 1 {
    let row = buckets[i]
    row.sort_by((a, b) => a.idx.compare(b.idx))
    merged_ptr[i] = col_idx.length()
    let mut k = 0
    while k < row.length() {
      let col = row[k].idx
      let mut acc = row[k].val
      let mut k2 = k + 1
      while k2 < row.length() && row[k2].idx == col {
        acc = acc + row[k2].val
        k2 = k2 + 1
      }
      col_idx.push(col)
      vals.push(acc)
      k = k2
    }
  }
  merged_ptr[self.nrows] = col_idx.length()
  @csr.Csr::new_unchecked(self.nrows, self.ncols, merged_ptr, col_idx, vals)
}

///|
/// Expand to a dense row-major array, summing duplicates.
pub fn[F : Add + Default] Coo::to_dense(self : Coo[F]) -> Array[F] {
  let dense : Array[F] = Array::make(
    self.nrows * self.ncols,
    Default::default(),
  )
  for k = 0; k < self.vals.length(); k = k + 1 {
    let at = self.rows[k] * self.ncols + self.cols[k]
    dense[at] = dense[at] + self.vals[k]
  }
  dense
}

///|
/// True when some coordinate appears more than once.
///
/// Worth checking before a conversion when the caller expects a clean matrix:
/// summing duplicates is silent, so an accidental repeat would otherwise go
/// unnoticed.
pub fn[F] Coo::has_duplicates(self : Coo[F]) -> Bool {
  let seen : Array[Int] = Array(capacity=self.vals.length())
  for k = 0; k < self.vals.length(); k = k + 1 {
    seen.push(self.rows[k] * self.ncols + self.cols[k])
  }
  seen.sort()
  for k = 1; k < seen.length(); k = k + 1 {
    if seen[k] == seen[k - 1] {
      return true
    }
  }
  false
}

///|
/// Number of coordinates that appear more than once.
pub fn[F] Coo::duplicate_count(self : Coo[F]) -> Int {
  let seen : Array[Int] = Array(capacity=self.vals.length())
  for k = 0; k < self.vals.length(); k = k + 1 {
    seen.push(self.rows[k] * self.ncols + self.cols[k])
  }
  seen.sort()
  let mut dups = 0
  for k = 1; k < seen.length(); k = k + 1 {
    if seen[k] == seen[k - 1] {
      dups = dups + 1
    }
  }
  dups
}

///|
/// Convert CSR back to COO, in row-major order.
///
/// Useful for feeding a matrix to a format writer or to a dense reference
/// implementation without going through a dense allocation.
pub fn[F] Coo::from_csr(a : @csr.Csr[F]) -> Coo[F] {
  let nrows = a.nrows()
  let ncols = a.ncols()
  let nnz = a.nnz()
  let row_ptr = a.row_ptr()
  let col_idx = a.col_idx()
  let vals = a.vals()
  let rows : Array[Int] = Array(capacity=nnz)
  let cols : Array[Int] = Array(capacity=nnz)
  let out_vals : Array[F] = Array(capacity=nnz)
  for i = 0; i < nrows; i = i + 1 {
    for k = row_ptr[i]; k < row_ptr[i + 1]; k = k + 1 {
      rows.push(i)
      cols.push(col_idx[k])
      out_vals.push(vals[k])
    }
  }
  { nrows, ncols, rows, cols, vals: out_vals, }
}
