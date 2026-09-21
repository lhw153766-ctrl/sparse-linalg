// Black-box tests for the CSC package: column access, the transposed product,
// and the conversions that must agree with the CSR implementations.

///|
fn raises(build : () -> Unit raise @err.SparseError) -> Bool {
  try {
    build()
    false
  } catch {
    _ => true
  }
}

// [[1, 0, 2],
//  [0, 3, 0]]
// CSC: col 0 -> row 0 (1.0) ; col 1 -> row 1 (3.0) ; col 2 -> row 0 (2.0)

///|
fn sample() -> @csc.Csc[Double] raise @err.SparseError {
  @csc.Csc::new(2, 3, [0, 1, 2, 3], [0, 1, 0], [1.0, 3.0, 2.0])
}

///|
test "shape and counts" {
  let a = sample()
  inspect(a.nrows(), content="2")
  inspect(a.ncols(), content="3")
  inspect(a.nnz(), content="3")
  inspect(a.col_nnz(0), content="1")
  inspect(a.col_nnz(2), content="1")
  debug_inspect(a.col_range(1), content="(1, 2)")
  inspect(a.max_col_nnz(), content="1")
  debug_inspect(a.col_ptr(), content="[0, 1, 2, 3]")
  debug_inspect(a.row_idx(), content="[0, 1, 0]")
}

///|
test "find, get, contains" {
  let a = sample()
  debug_inspect(a.get(0, 0), content="Some(1)")
  debug_inspect(a.get(1, 1), content="Some(3)")
  debug_inspect(a.get(0, 2), content="Some(2)")
  debug_inspect(a.get(1, 0), content="None")
  inspect(a.contains(1, 1), content="true")
  inspect(a.contains(0, 1), content="false")
  inspect(a.get_or(0, 1, -1.0), content="-1")
  inspect(a.get_or(0, 2, -1.0), content="2")
}

///|
test "spmv agrees with the CSR result" {
  let a = sample()
  // [1,0,2; 0,3,0] * [1,2,3] = [7, 6]
  debug_inspect(a.spmv([1.0, 2.0, 3.0]), content="[7, 6]")
}

///|
test "spmv_transpose computes A^T x" {
  let a = sample()
  // A^T = [1,0; 0,3; 2,0]; A^T * [1,1] = [1, 3, 2]
  debug_inspect(a.spmv_transpose([1.0, 1.0]), content="[1, 3, 2]")
}

///|
test "spmv and spmv_transpose reject wrong lengths" {
  let a = sample()
  inspect(
    raises(() => {
      let _ = a.spmv([1.0])
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = a.spmv_transpose([1.0, 2.0, 3.0])
    }),
    content="true",
  )
}

///|
test "diagonal" {
  let a = sample()
  debug_inspect(a.diagonal(), content="[1, 3]")
}

///|
test "to_dense matches the row-major layout" {
  let a = sample()
  debug_inspect(a.to_dense(), content="[1, 0, 2, 0, 3, 0]")
}

///|
test "scale" {
  let a = sample()
  debug_inspect(a.scale(2.0).vals(), content="[2, 6, 4]")
  debug_inspect(a.vals(), content="[1, 3, 2]")
  let e : @csc.Csc[Double] = @csc.Csc::new(1, 1, [0, 0], [], [])
  inspect(e.scale(3.0).nnz(), content="0")
}

///|
test "is_symmetric" {
  let sym : @csc.Csc[Double] = @csc.Csc::new(2, 2, [0, 2, 4], [0, 1, 0, 1], [
    2.0, 1.0, 1.0, 3.0,
  ])
  inspect(sym.is_symmetric(), content="true")
  let asym : @csc.Csc[Double] = @csc.Csc::new(2, 2, [0, 2, 4], [0, 1, 0, 1], [
    2.0, 1.0, 9.0, 3.0,
  ])
  inspect(asym.is_symmetric(), content="false")
  let rect : @csc.Csc[Double] = @csc.Csc::new(1, 2, [0, 1, 1], [0], [1.0])
  inspect(rect.is_symmetric(), content="false")
}

///|
test "new rejects malformed structure" {
  // col_ptr wrong length
  inspect(
    raises(() => {
      let _ : @csc.Csc[Double] = @csc.Csc::new(2, 2, [0, 1], [0], [1.0])
    }),
    content="true",
  )
  // col_ptr must start at 0
  inspect(
    raises(() => {
      let _ : @csc.Csc[Double] = @csc.Csc::new(1, 1, [1, 1], [], [])
    }),
    content="true",
  )
  // col_ptr must end at nnz
  inspect(
    raises(() => {
      let _ : @csc.Csc[Double] = @csc.Csc::new(1, 1, [0, 2], [0], [1.0])
    }),
    content="true",
  )
  // row_idx and vals lengths must agree
  inspect(
    raises(() => {
      let _ : @csc.Csc[Double] = @csc.Csc::new(2, 1, [0, 2], [0], [1.0, 2.0])
    }),
    content="true",
  )
  // row out of range
  inspect(
    raises(() => {
      let _ : @csc.Csc[Double] = @csc.Csc::new(2, 1, [0, 1], [7], [1.0])
    }),
    content="true",
  )
  // duplicated row inside a column
  inspect(
    raises(() => {
      let _ : @csc.Csc[Double] = @csc.Csc::new(2, 1, [0, 2], [1, 1], [1.0, 2.0])
    }),
    content="true",
  )
  // descending rows inside a column
  inspect(
    raises(() => {
      let _ : @csc.Csc[Double] = @csc.Csc::new(2, 1, [0, 2], [1, 0], [1.0, 2.0])
    }),
    content="true",
  )
}

///|
test "CSR -> CSC -> CSR is the identity" {
  let csr : @csr.Csr[Double] = @csr.Csr::new(
    3,
    4,
    [0, 2, 3, 5],
    [0, 3, 1, 0, 2],
    [1.0, 2.0, 3.0, 4.0, 5.0],
  )
  let csc = @csc.Csc::from_csr(csr)
  inspect(csc.nrows(), content="3")
  inspect(csc.ncols(), content="4")
  inspect(csc.nnz(), content="5")
  let back = csc.to_csr()
  debug_inspect(back.row_ptr(), content="[0, 2, 3, 5]")
  debug_inspect(back.col_idx(), content="[0, 3, 1, 0, 2]")
  debug_inspect(back.vals(), content="[1, 2, 3, 4, 5]")
}

///|
test "CSR -> CSC -> CSR preserves the dense matrix on a rectangular case" {
  let csr : @csr.Csr[Double] = @csr.Csr::new(4, 2, [0, 1, 1, 2, 3], [1, 0, 1], [
    1.5, 2.5, 3.5,
  ])
  let dense_before = csr.to_dense()
  let dense_after = @csc.Csc::from_csr(csr).to_csr().to_dense()
  debug_inspect(dense_after, content="[0, 1.5, 0, 0, 2.5, 0, 0, 3.5]")
  debug_inspect(dense_before, content="[0, 1.5, 0, 0, 2.5, 0, 0, 3.5]")
}

///|
test "CSC spmv equals CSR spmv" {
  let csr : @csr.Csr[Double] = @csr.Csr::new(
    3,
    3,
    [0, 2, 3, 5],
    [0, 2, 1, 0, 2],
    [1.0, 2.0, 3.0, 4.0, 5.0],
  )
  let csc = @csc.Csc::from_csr(csr)
  let x = [1.0, -1.0, 2.0]
  debug_inspect(csc.spmv(x), content="[5, -3, 14]")
  debug_inspect(csr.spmv(x), content="[5, -3, 14]")
}

///|
test "spmv_transpose in CSC equals transposing then multiplying in CSR" {
  let csr : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  let csc = @csc.Csc::from_csr(csr)
  // A^T x with x = [1, 1] -> [1, 3, 2]
  debug_inspect(csc.spmv_transpose([1.0, 1.0]), content="[1, 3, 2]")
  // Cross-check against the dense transpose built by hand.
  let at : @csr.Csr[Double] = @csr.Csr::new(3, 2, [0, 1, 2, 3], [0, 1, 0], [
    1.0, 3.0, 2.0,
  ])
  debug_inspect(at.spmv([1.0, 1.0]), content="[1, 3, 2]")
}

///|
test "from_coo sums duplicates and sorts rows" {
  // (1,0)=1 ; (1,0)=2 ; (0,0)=5  -> column 0 holds rows 0 and 1, with row 1 = 3
  let coo : @coo.Coo[Double] = @coo.Coo::new(2, 1, [1, 1, 0], [0, 0, 0], [
    1.0, 2.0, 5.0,
  ])
  let csc = @csc.Csc::from_coo(coo)
  inspect(csc.nnz(), content="2")
  debug_inspect(csc.col_ptr(), content="[0, 2]")
  debug_inspect(csc.row_idx(), content="[0, 1]")
  debug_inspect(csc.vals(), content="[5, 3]")
}

///|
test "from_coo on an empty matrix" {
  let coo : @coo.Coo[Double] = @coo.Coo::empty(2, 3)
  let csc = @csc.Csc::from_coo(coo)
  inspect(csc.nnz(), content="0")
  debug_inspect(csc.col_ptr(), content="[0, 0, 0, 0]")
}

///|
test "sum_vals and Int64 exactness" {
  let a : @csc.Csc[Int64] = @csc.Csc::new(2, 1, [0, 2], [0, 1], [10L, 20L])
  inspect(a.sum_vals(), content="30")
  debug_inspect(a.spmv([3L]), content="[30, 60]")
}

///|
test "accessors reject out-of-range columns" {
  let a = sample()
  inspect(
    raises(() => {
      let _ = a.col_nnz(3)
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = a.col_range(-1)
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = a.find(0, 3)
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = a.find(2, 0)
    }),
    content="true",
  )
}
