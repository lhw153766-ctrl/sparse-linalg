// Conversions between the three storage layouts.
//
// The CSR <-> CSC pair is a counting sort: count how many entries land in each
// row (or column), prefix-sum the counts into the pointer array, then scatter.
// That is `O(nnz)` and needs no comparison sort, because the destination order
// is determined by the pointer array rather than by ordering the values.

///|
/// Convert CSR to CSC in `O(nnz)`.
///
/// This is the transpose in disguise: the entries of row `i` of `A` become the
/// entries of column `i` of `A^T`, and CSC stores columns. Feeding the result
/// back through `Csc::to_csr` therefore transposes twice, which the round-trip
/// tests check.
pub fn[F] Csc::from_csr(a : @csr.Csr[F]) -> Csc[F] {
  let nrows = a.nrows()
  let ncols = a.ncols()
  let nnz = a.nnz()
  let row_ptr = a.row_ptr()
  let col_idx = a.col_idx()
  let vals = a.vals()
  // count entries per column
  let col_ptr : Array[Int] = Array::make(ncols + 1, 0)
  for k = 0; k < nnz; k = k + 1 {
    col_ptr[col_idx[k] + 1] = col_ptr[col_idx[k] + 1] + 1
  }
  for j = 0; j < ncols; j = j + 1 {
    col_ptr[j + 1] = col_ptr[j + 1] + col_ptr[j]
  }
  // scatter: walk rows in order, so each column receives ascending rows
  let next : Array[Int] = col_ptr.copy()
  let row_idx : Array[Int] = Array::make(nnz, 0)
  let out_vals : Array[F] = Array::make(nnz, vals[0])
  for i = 0; i < nrows; i = i + 1 {
    for k = row_ptr[i]; k < row_ptr[i + 1]; k = k + 1 {
      let j = col_idx[k]
      let at = next[j]
      row_idx[at] = i
      out_vals[at] = vals[k]
      next[j] = at + 1
    }
  }
  Csc::new_unchecked(nrows, ncols, col_ptr, row_idx, out_vals)
}

///|
/// Convert CSC to CSR in `O(nnz)`. The mirror of `Csc::from_csr`.
pub fn[F] Csc::to_csr(self : Csc[F]) -> @csr.Csr[F] {
  let nrows = self.nrows
  let ncols = self.ncols
  let nnz = self.vals.length()
  let col_ptr = self.col_ptr
  let row_idx = self.row_idx
  let vals = self.vals
  let row_ptr : Array[Int] = Array::make(nrows + 1, 0)
  for k = 0; k < nnz; k = k + 1 {
    row_ptr[row_idx[k] + 1] = row_ptr[row_idx[k] + 1] + 1
  }
  for i = 0; i < nrows; i = i + 1 {
    row_ptr[i + 1] = row_ptr[i + 1] + row_ptr[i]
  }
  let next : Array[Int] = row_ptr.copy()
  let col_idx : Array[Int] = Array::make(nnz, 0)
  let out_vals : Array[F] = Array::make(nnz, vals[0])
  for j = 0; j < ncols; j = j + 1 {
    for k = col_ptr[j]; k < col_ptr[j + 1]; k = k + 1 {
      let i = row_idx[k]
      let at = next[i]
      col_idx[at] = j
      out_vals[at] = vals[k]
      next[i] = at + 1
    }
  }
  @csr.Csr::new_unchecked(nrows, ncols, row_ptr, col_idx, out_vals)
}

///|
/// Convert a COO matrix to CSC, summing duplicate coordinates.
///
/// Unlike the CSR/CSC pair this needs a comparison sort per column, because
/// duplicates have to be brought adjacent before they can be summed.
pub fn[F : Add] Csc::from_coo(a : @coo.Coo[F]) -> Csc[F] {
  let nrows = a.nrows()
  let ncols = a.ncols()
  let rows = a.rows()
  let cols = a.cols()
  let vals = a.vals()
  let nnz = vals.length()
  let col_ptr : Array[Int] = Array::make(ncols + 1, 0)
  for k = 0; k < nnz; k = k + 1 {
    col_ptr[cols[k] + 1] = col_ptr[cols[k] + 1] + 1
  }
  for j = 0; j < ncols; j = j + 1 {
    col_ptr[j + 1] = col_ptr[j + 1] + col_ptr[j]
  }
  let buckets : Array[Array[Slot[F]]] = Array::make(ncols, [])
  for j = 0; j < ncols; j = j + 1 {
    buckets[j] = Array(capacity=col_ptr[j + 1] - col_ptr[j])
  }
  for k = 0; k < nnz; k = k + 1 {
    buckets[cols[k]].push({ idx: rows[k], val: vals[k], })
  }
  let row_idx : Array[Int] = Array(capacity=nnz)
  let out_vals : Array[F] = Array(capacity=nnz)
  let merged_ptr : Array[Int] = Array::make(ncols + 1, 0)
  for j = 0; j < ncols; j = j + 1 {
    let col = buckets[j]
    col.sort_by((x, y) => x.idx.compare(y.idx))
    merged_ptr[j] = row_idx.length()
    let mut k = 0
    while k < col.length() {
      let r = col[k].idx
      let mut acc = col[k].val
      let mut k2 = k + 1
      while k2 < col.length() && col[k2].idx == r {
        acc = acc + col[k2].val
        k2 = k2 + 1
      }
      row_idx.push(r)
      out_vals.push(acc)
      k = k2
    }
  }
  merged_ptr[ncols] = row_idx.length()
  Csc::new_unchecked(nrows, ncols, merged_ptr, row_idx, out_vals)
}

// Sort key for assembling a row or column. Mirrors the private `Slot` in the
// coo package; equal indices are summed, so the relative order of duplicates
// does not matter.

///|
priv struct Slot[F] {
  idx : Int
  val : F
}
