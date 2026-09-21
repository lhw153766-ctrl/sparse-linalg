// Black-box tests for the CSR package. The matrices here are small enough to
// check by hand, which is the point: the differential tests against a dense
// oracle live in the module-level `differential` package.

///|
fn raises(build : () -> Unit raise @err.SparseError) -> Bool {
  try {
    build()
    false
  } catch {
    _ => true
  }
}

///|
test "shape and counts" {
  // [[1, 0, 2],
  //  [0, 3, 0]]
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  inspect(a.nrows(), content="2")
  inspect(a.ncols(), content="3")
  inspect(a.nnz(), content="3")
  inspect(a.row_nnz(0), content="2")
  inspect(a.row_nnz(1), content="1")
  debug_inspect(a.row_nnz_all(), content="[2, 1]")
  inspect(a.max_row_nnz(), content="2")
  debug_inspect(a.row_range(1), content="(2, 3)")
  inspect(a.shape().is_square(), content="false")
  inspect(a.shape().dense_len(), content="6")
}

///|
test "find, get, contains" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  debug_inspect(a.get(0, 0), content="Some(1)")
  debug_inspect(a.get(0, 2), content="Some(2)")
  debug_inspect(a.get(0, 1), content="None")
  inspect(a.contains(1, 1), content="true")
  inspect(a.contains(0, 1), content="false")
  inspect(a.get_or(0, 1, 0.0), content="0")
  inspect(a.get_or(1, 1, -1.0), content="3")
}

///|
test "spmv matches hand computation" {
  // [[1, 0, 2],
  //  [0, 3, 0]]  * [1, 2, 3] = [1*1 + 2*3, 3*2] = [7, 6]
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  debug_inspect(a.spmv([1.0, 2.0, 3.0]), content="[7, 6]")
}

///|
test "spmv on an all-zero matrix" {
  let z : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 0, 0, 0], [], [])
  inspect(z.nnz(), content="0")
  debug_inspect(z.spmv([1.0, 2.0, 3.0]), content="[0, 0, 0]")
  inspect(z.max_row_nnz(), content="0")
}

///|
test "spmv rejects a short vector" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  inspect(
    raises(() => {
      let _ = a.spmv([1.0, 2.0])
    }),
    content="true",
  )
}

///|
test "spmv_add applies alpha and beta" {
  // y = 2*A*x + 0.5*y
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [1.0, 4.0])
  let y = [1.0, 1.0]
  a.spmv_add(2.0, [1.0, 1.0], 0.5, y)
  debug_inspect(y, content="[2.5, 8.5]")
}

///|
test "diagonal includes implicit zeros" {
  let a : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 1, 1, 2], [0, 2], [
    5.0, 7.0,
  ])
  debug_inspect(a.diagonal(), content="[5, 0, 7]")
}

///|
test "is_symmetric" {
  let sym : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 2, 4], [0, 1, 0, 1], [
    2.0, 1.0, 1.0, 3.0,
  ])
  inspect(sym.is_symmetric(), content="true")
  let asym : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 2, 4], [0, 1, 0, 1], [
    2.0, 1.0, 9.0, 3.0,
  ])
  inspect(asym.is_symmetric(), content="false")
  // Structurally symmetric but with a missing mirror entry.
  let hole : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 2, 2], [0, 1], [
    2.0, 1.0,
  ])
  inspect(hole.is_symmetric(), content="false")
  // Non-square is never symmetric.
  let rect : @csr.Csr[Double] = @csr.Csr::new(1, 2, [0, 1], [0], [1.0])
  inspect(rect.is_symmetric(), content="false")
}

///|
test "to_dense round trip" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  debug_inspect(a.to_dense(), content="[1, 0, 2, 0, 3, 0]")
}

///|
test "scale, including the empty case" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [1.0, -2.0])
  let b = a.scale(3.0)
  debug_inspect(b.vals(), content="[3, -6]")
  debug_inspect(a.vals(), content="[1, -2]")
  let empty : @csr.Csr[Double] = @csr.Csr::new(1, 1, [0, 0], [], [])
  inspect(empty.scale(2.0).nnz(), content="0")
}

///|
test "sum_vals keeps exactness for Int64" {
  let a : @csr.Csr[Int64] = @csr.Csr::new(1, 3, [0, 3], [0, 1, 2], [3L, 4L, 5L])
  inspect(a.sum_vals(), content="12")
  debug_inspect(a.spmv([10L, 100L, 1000L]), content="[5430]")
}

///|
test "Float works as well as Double" {
  let a : @csr.Csr[Float] = @csr.Csr::new(1, 2, [0, 2], [0, 1], [1.5, 2.5])
  debug_inspect(a.spmv([2.0, 2.0]), content="[8]")
}

///|
test "new rejects malformed structure" {
  // row_ptr must have nrows + 1 entries.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1], [0], [1.0])
    }),
    content="true",
  )
  // row_ptr must start at 0.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(1, 1, [1, 1], [], [])
    }),
    content="true",
  )
  // row_ptr must end at nnz.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(1, 1, [0, 2], [0], [1.0])
    }),
    content="true",
  )
  // row_ptr must not decrease.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(2, 1, [0, 1, 0], [0], [1.0])
    }),
    content="true",
  )
  // col_idx and vals must agree in length.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(1, 2, [0, 2], [0], [1.0, 2.0])
    }),
    content="true",
  )
  // Column index out of range.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(1, 2, [0, 1], [5], [1.0])
    }),
    content="true",
  )
  // Negative column index.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(1, 2, [0, 1], [-1], [1.0])
    }),
    content="true",
  )
  // Duplicated column in one row.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(1, 2, [0, 2], [1, 1], [1.0, 2.0])
    }),
    content="true",
  )
  // Descending columns in one row.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(1, 2, [0, 2], [1, 0], [1.0, 2.0])
    }),
    content="true",
  )
  // Negative shape.
  inspect(
    raises(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(-1, 1, [], [], [])
    }),
    content="true",
  )
}

///|
test "accessors reject out-of-range indices" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 1], [0], [1.0])
  inspect(
    raises(() => {
      let _ = a.row_nnz(5)
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = a.row_nnz(-1)
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = a.row_range(2)
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = a.find(0, 9)
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = a.get(0, -1)
    }),
    content="true",
  )
}

///|
test "new_unchecked skips validation for builder output" {
  let a : @csr.Csr[Double] = @csr.Csr::new_unchecked(1, 2, [0, 2], [0, 1], [
    1.0, 2.0,
  ])
  inspect(a.nnz(), content="2")
  debug_inspect(a.spmv([3.0, 4.0]), content="[11]")
}
