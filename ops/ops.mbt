// Structural operations on sparse matrices: addition, subtraction,
// transposition, and the two products.
//
// Everything here returns CSR, which is the layout the solvers consume. Where a
// result is naturally produced column-wise (transposition, for instance) the
// work is done in CSC and converted once at the end, rather than being built
// row-wise with a scatter that needs random writes.

///|
/// Transpose a CSR matrix.
///
/// A CSC matrix whose column pointer array is `A`'s row pointer array *is*
/// `A^T`: column `i` of the result holds exactly the entries of row `i` of `A`,
/// with the row indices of the result being `A`'s column indices. So the
/// transpose is a reinterpretation of the arrays, not a scatter, and the
/// conversion back to CSR is the only real work.
pub fn[F] transpose(a : @csr.Csr[F]) -> @csr.Csr[F] {
  let as_csc : @csc.Csc[F] = @csc.Csc::new_unchecked(
    a.ncols(),
    a.nrows(),
    a.row_ptr().copy(),
    a.col_idx().copy(),
    a.vals().copy(),
  )
  as_csc.to_csr()
}

///|
/// `A + B`.
///
/// The result's pattern is the union of the two patterns: a coordinate stored
/// in only one operand is kept, and a coordinate stored in both is added. An
/// exact cancellation is dropped rather than stored as an explicit zero, so the
/// result carries no structure that contributes nothing.
pub fn[F : Add + Default + Eq] add(
  a : @csr.Csr[F],
  b : @csr.Csr[F],
) -> @csr.Csr[F] raise @err.SparseError {
  check_same_shape("add", a, b)
  let nrows = a.nrows()
  let arp = a.row_ptr()
  let acol = a.col_idx()
  let aval = a.vals()
  let brp = b.row_ptr()
  let bcol = b.col_idx()
  let bval = b.vals()
  let row_ptr : Array[Int] = Array(capacity=nrows + 1)
  let out_col : Array[Int] = Array(capacity=a.nnz() + b.nnz())
  let out_val : Array[F] = Array(capacity=a.nnz() + b.nnz())
  row_ptr.push(0)
  for i = 0; i < nrows; i = i + 1 {
    let mut pa = arp[i]
    let mut pb = brp[i]
    while pa < arp[i + 1] || pb < brp[i + 1] {
      let a_done = pa >= arp[i + 1]
      let b_done = pb >= brp[i + 1]
      if !a_done && (b_done || acol[pa] < bcol[pb]) {
        out_col.push(acol[pa])
        out_val.push(aval[pa])
        pa = pa + 1
      } else if !b_done && (a_done || bcol[pb] < acol[pa]) {
        out_col.push(bcol[pb])
        out_val.push(bval[pb])
        pb = pb + 1
      } else {
        let combined = aval[pa] + bval[pb]
        if combined != Default::default() {
          out_col.push(acol[pa])
          out_val.push(combined)
        }
        pa = pa + 1
        pb = pb + 1
      }
    }
    row_ptr.push(out_col.length())
  }
  @csr.Csr::new_unchecked(nrows, a.ncols(), row_ptr, out_col, out_val)
}

///|
/// `A - B`. The pattern rules match `add`.
pub fn[F : Sub + Default + Eq] sub(
  a : @csr.Csr[F],
  b : @csr.Csr[F],
) -> @csr.Csr[F] raise @err.SparseError {
  check_same_shape("sub", a, b)
  let nrows = a.nrows()
  let arp = a.row_ptr()
  let acol = a.col_idx()
  let aval = a.vals()
  let brp = b.row_ptr()
  let bcol = b.col_idx()
  let bval = b.vals()
  let row_ptr : Array[Int] = Array(capacity=nrows + 1)
  let out_col : Array[Int] = Array(capacity=a.nnz() + b.nnz())
  let out_val : Array[F] = Array(capacity=a.nnz() + b.nnz())
  row_ptr.push(0)
  for i = 0; i < nrows; i = i + 1 {
    let mut pa = arp[i]
    let mut pb = brp[i]
    while pa < arp[i + 1] || pb < brp[i + 1] {
      let a_done = pa >= arp[i + 1]
      let b_done = pb >= brp[i + 1]
      if !a_done && (b_done || acol[pa] < bcol[pb]) {
        out_col.push(acol[pa])
        out_val.push(aval[pa])
        pa = pa + 1
      } else if !b_done && (a_done || bcol[pb] < acol[pa]) {
        // B contributes with the opposite sign; -0.0 is fine here.
        out_col.push(bcol[pb])
        out_val.push(Default::default() - bval[pb])
        pb = pb + 1
      } else {
        let combined = aval[pa] - bval[pb]
        if combined != Default::default() {
          out_col.push(acol[pa])
          out_val.push(combined)
        }
        pa = pa + 1
        pb = pb + 1
      }
    }
    row_ptr.push(out_col.length())
  }
  @csr.Csr::new_unchecked(nrows, a.ncols(), row_ptr, out_col, out_val)
}

///|
fn[F] check_same_shape(
  who : String,
  a : @csr.Csr[F],
  b : @csr.Csr[F],
) -> Unit raise @err.SparseError {
  if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
    raise @err.SparseError::DimensionMismatch(
      who +
      ": shapes differ, " +
      a.nrows().to_string() +
      "x" +
      a.ncols().to_string() +
      " and " +
      b.nrows().to_string() +
      "x" +
      b.ncols().to_string(),
    )
  }
}

///|
/// `A * B` where `A` is sparse and `B` is dense.
///
/// `b` is row-major with `ncols` columns; the result is row-major with the same
/// column count. Used to apply a sparse operator to several right-hand sides at
/// once, and to advance a block of vectors in a block Krylov method.
pub fn[F : Add + Mul + Default] spmm(
  a : @csr.Csr[F],
  b : Array[F],
  b_ncols : Int,
) -> Array[F] raise @err.SparseError {
  if b_ncols < 0 || b.length() != a.ncols() * b_ncols {
    raise @err.SparseError::DimensionMismatch(
      "spmm: B holds " +
      b.length().to_string() +
      " entries, expected " +
      (a.ncols() * b_ncols).to_string(),
    )
  }
  let nrows = a.nrows()
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let va = a.vals()
  let out : Array[F] = Array::make(nrows * b_ncols, Default::default())
  for i = 0; i < nrows; i = i + 1 {
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      let aik = va[k]
      let src = ci[k] * b_ncols
      let dst = i * b_ncols
      for j = 0; j < b_ncols; j = j + 1 {
        out[dst + j] = out[dst + j] + aik * b[src + j]
      }
    }
  }
  out
}

///|
/// `A * B` where both are sparse.
///
/// Each row of the product is accumulated into a dense workspace the width of a
/// row of `B`, with a list of the columns actually touched so that only those
/// are scanned afterwards. Numerical zeros are dropped, which keeps the pattern
/// from growing without bound as products are chained.
pub fn[F : Add + Mul + Default + Eq] spgemm(
  a : @csr.Csr[F],
  b : @csr.Csr[F],
) -> @csr.Csr[F] raise @err.SparseError {
  if a.ncols() != b.nrows() {
    raise @err.SparseError::DimensionMismatch(
      "spgemm: inner dimensions differ, A is " +
      a.nrows().to_string() +
      "x" +
      a.ncols().to_string() +
      " and B is " +
      b.nrows().to_string() +
      "x" +
      b.ncols().to_string(),
    )
  }
  let nrows = a.nrows()
  let ncols = b.ncols()
  let arp = a.row_ptr()
  let acol = a.col_idx()
  let aval = a.vals()
  let brp = b.row_ptr()
  let bcol = b.col_idx()
  let bval = b.vals()
  let row_ptr : Array[Int] = Array(capacity=nrows + 1)
  let out_col : Array[Int] = Array(capacity=0)
  let out_val : Array[F] = Array(capacity=0)
  row_ptr.push(0)
  // Workspace reused across rows. `touched` records which columns were written
  // so the reset at the end of each row costs O(touched), not O(ncols).
  let acc : Array[F] = Array::make(ncols, Default::default())
  let touched : Array[Int] = Array(capacity=ncols)
  for i = 0; i < nrows; i = i + 1 {
    touched.clear()
    for k = arp[i]; k < arp[i + 1]; k = k + 1 {
      let aik = aval[k]
      let brow = acol[k]
      for m = brp[brow]; m < brp[brow + 1]; m = m + 1 {
        let j = bcol[m]
        if acc[j] == Default::default() {
          touched.push(j)
        }
        acc[j] = acc[j] + aik * bval[m]
      }
    }
    touched.sort()
    for j in touched {
      let v = acc[j]
      if v != Default::default() {
        out_col.push(j)
        out_val.push(v)
      }
      acc[j] = Default::default()
    }
    row_ptr.push(out_col.length())
  }
  @csr.Csr::new_unchecked(nrows, ncols, row_ptr, out_col, out_val)
}

///|
/// The `n x n` identity, with `one` on the diagonal.
///
/// The diagonal value is a parameter rather than a constant because the library
/// is generic over the scalar type and there is no single notion of "one" that
/// covers integers and floating point alike.
pub fn[F] identity(n : Int, one : F) -> @csr.Csr[F] raise @err.SparseError {
  if n < 0 {
    raise @err.SparseError::DimensionMismatch(
      "identity: negative size " + n.to_string(),
    )
  }
  let row_ptr : Array[Int] = Array::make(n + 1, 0)
  let col_idx : Array[Int] = Array::make(n, 0)
  let vals : Array[F] = Array::make(n, one)
  for i = 0; i < n; i = i + 1 {
    row_ptr[i] = i
    col_idx[i] = i
  }
  row_ptr[n] = n
  @csr.Csr::new_unchecked(n, n, row_ptr, col_idx, vals)
}

///|
/// Build a sparse matrix from a dense row-major array.
///
/// `is_stored` decides which entries become structural non-zeros; passing a
/// tolerance-based predicate is the usual choice, since floating-point results
/// are rarely exactly zero.
pub fn[F] from_dense(
  dense : Array[F],
  nrows : Int,
  ncols : Int,
  is_stored : (F) -> Bool,
) -> @csr.Csr[F] raise @err.SparseError {
  if nrows < 0 || ncols < 0 || dense.length() != nrows * ncols {
    raise @err.SparseError::DimensionMismatch(
      "from_dense: expected " +
      (nrows * ncols).to_string() +
      " entries, got " +
      dense.length().to_string(),
    )
  }
  let row_ptr : Array[Int] = Array::make(nrows + 1, 0)
  let col_idx : Array[Int] = Array(capacity=dense.length())
  let vals : Array[F] = Array(capacity=dense.length())
  for i = 0; i < nrows; i = i + 1 {
    for j = 0; j < ncols; j = j + 1 {
      let v = dense[i * ncols + j]
      if is_stored(v) {
        col_idx.push(j)
        vals.push(v)
      }
    }
    row_ptr[i + 1] = col_idx.length()
  }
  @csr.Csr::new_unchecked(nrows, ncols, row_ptr, col_idx, vals)
}

///|
/// Remove explicitly stored zeros.
///
/// Useful after a subtraction or a product that left structure contributing
/// nothing, when the pattern itself is what matters — a reordering or a
/// factorisation, for example.
pub fn[F : Default + Eq] drop_zeros(a : @csr.Csr[F]) -> @csr.Csr[F] {
  let nrows = a.nrows()
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let va = a.vals()
  let row_ptr : Array[Int] = Array::make(nrows + 1, 0)
  let out_col : Array[Int] = Array(capacity=a.nnz())
  let out_val : Array[F] = Array(capacity=a.nnz())
  for i = 0; i < nrows; i = i + 1 {
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      if va[k] != Default::default() {
        out_col.push(ci[k])
        out_val.push(va[k])
      }
    }
    row_ptr[i + 1] = out_col.length()
  }
  @csr.Csr::new_unchecked(nrows, a.ncols(), row_ptr, out_col, out_val)
}

///|
/// Scale every stored value by `alpha`, keeping the pattern.
pub fn[F : Mul] scale(a : @csr.Csr[F], alpha : F) -> @csr.Csr[F] {
  a.scale(alpha)
}
