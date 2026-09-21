// Black-box tests for the structural operations. Each result is checked both
// as a dense expansion (so the numbers are readable) and as a structure
// (so a right answer stored in the wrong pattern is still caught).

///|
test "add unions the patterns" {
  // A = [1, 0 ; 0, 2]   B = [0, 3 ; 0, 0]
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0])
  let b : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 1], [1], [3.0])
  let c = @ops.add(a, b)
  debug_inspect(c.to_dense(), content="[1, 3, 0, 2]")
  debug_inspect(c.row_ptr(), content="[0, 2, 3]")
  debug_inspect(c.col_idx(), content="[0, 1, 1]")
}

///|
test "add drops an exact cancellation" {
  let a : @csr.Csr[Double] = @csr.Csr::new(1, 2, [0, 1], [0], [5.0])
  let b : @csr.Csr[Double] = @csr.Csr::new(1, 2, [0, 1], [0], [-5.0])
  let c = @ops.add(a, b)
  inspect(c.nnz(), content="0")
  debug_inspect(c.to_dense(), content="[0, 0]")
}

///|
test "add on disjoint and empty operands" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0])
  let empty : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 0, 0], [], [])
  debug_inspect(@ops.add(a, empty).to_dense(), content="[1, 0, 0, 2]")
  debug_inspect(@ops.add(empty, a).to_dense(), content="[1, 0, 0, 2]")
  inspect(@ops.add(empty, empty).nnz(), content="0")
}

///|
test "add rejects mismatched shapes" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 1], [0], [1.0])
  let b : @csr.Csr[Double] = @csr.Csr::new(3, 2, [0, 0, 0, 0], [], [])
  let raised = try {
    let _ = @ops.add(a, b)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "sub subtracts and keeps the union pattern" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0])
  let b : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 1], [1], [3.0])
  let c = @ops.sub(a, b)
  // [1, -3 ; 0, 2]
  debug_inspect(c.to_dense(), content="[1, -3, 0, 2]")
}

///|
test "sub then add is the identity" {
  let a : @csr.Csr[Double] = @csr.Csr::new(
    3,
    3,
    [0, 2, 3, 5],
    [0, 2, 1, 0, 2],
    [1.0, 2.0, 3.0, 4.0, 5.0],
  )
  let b : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 1, 3, 4], [1, 0, 2, 1], [
    7.0, 8.0, 9.0, 1.0,
  ])
  let round_trip = @ops.add(@ops.sub(a, b), b)
  debug_inspect(round_trip.to_dense(), content="[1, 0, 2, 0, 3, 0, 4, 0, 5]")
}

///|
test "transpose of a rectangular matrix" {
  // A = [1, 2, 3 ; 0, 4, 0]  ->  A^T = [1, 0 ; 2, 4 ; 3, 0]
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 3, 4], [0, 1, 2, 1], [
    1.0, 2.0, 3.0, 4.0,
  ])
  let t = @ops.transpose(a)
  inspect(t.nrows(), content="3")
  inspect(t.ncols(), content="2")
  debug_inspect(t.to_dense(), content="[1, 0, 2, 4, 3, 0]")
}

///|
test "transpose twice is the identity" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 3, 4], [0, 1, 2, 1], [
    1.0, 2.0, 3.0, 4.0,
  ])
  let tt = @ops.transpose(@ops.transpose(a))
  debug_inspect(tt.row_ptr(), content="[0, 3, 4]")
  debug_inspect(tt.col_idx(), content="[0, 1, 2, 1]")
  debug_inspect(tt.vals(), content="[1, 2, 3, 4]")
}

///|
test "transpose of the empty matrix" {
  let e : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 0, 0], [], [])
  let t = @ops.transpose(e)
  inspect(t.nrows(), content="3")
  inspect(t.ncols(), content="2")
  inspect(t.nnz(), content="0")
}

///|
test "transpose preserves symmetry for a symmetric matrix" {
  let s : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 2, 4], [0, 1, 0, 1], [
    2.0, 1.0, 1.0, 3.0,
  ])
  inspect(@ops.transpose(s).is_symmetric(), content="true")
}

///|
test "spmm against a two-column dense block" {
  // A = [1, 0 ; 0, 2], B = [[1, 2], [3, 4]] row-major -> [[1,2],[6,8]]
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0])
  let out = @ops.spmm(a, [1.0, 2.0, 3.0, 4.0], 2)
  debug_inspect(out, content="[1, 2, 6, 8]")
}

///|
test "spmm with a single column equals spmv" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  let x = [1.0, 2.0, 3.0]
  debug_inspect(@ops.spmm(a, x, 1), content="[7, 6]")
  debug_inspect(a.spmv(x), content="[7, 6]")
}

///|
test "spmm rejects a mis-sized dense operand" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  let raised = try {
    let _ = @ops.spmm(a, [1.0, 2.0, 3.0], 2)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "spgemm against a hand-computed product" {
  // A = [1, 2 ; 0, 3], B = [0, 1 ; 4, 0]
  // A*B = [1*0+2*4, 1*1+2*0 ; 0*0+3*4, 0*1+3*0] = [8, 1 ; 12, 0]
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 2, 3], [0, 1, 1], [
    1.0, 2.0, 3.0,
  ])
  let b : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [1, 0], [1.0, 4.0])
  let c = @ops.spgemm(a, b)
  debug_inspect(c.to_dense(), content="[8, 1, 12, 0]")
  // The zero in the bottom-right corner must not be stored.
  inspect(c.nnz(), content="3")
}

///|
test "spgemm with the identity leaves the matrix alone" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  let i3 = @ops.identity(3, 1.0)
  let c = @ops.spgemm(a, i3)
  debug_inspect(c.to_dense(), content="[1, 0, 2, 0, 3, 0]")
}

///|
test "spgemm of empty operands" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 0, 0], [], [])
  let b : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 0, 0], [], [])
  inspect(@ops.spgemm(a, b).nnz(), content="0")
}

///|
test "spgemm rejects a dimension mismatch" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 0, 0], [], [])
  let b : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 0, 0], [], [])
  let raised = try {
    let _ = @ops.spgemm(a, b)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "identity" {
  let i = @ops.identity(3, 1.0)
  debug_inspect(i.to_dense(), content="[1, 0, 0, 0, 1, 0, 0, 0, 1]")
  inspect(i.nnz(), content="3")
  inspect(i.is_symmetric(), content="true")
  let raised = try {
    let _ = @ops.identity(-1, 1.0)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "from_dense keeps only the entries the predicate accepts" {
  let a = @ops.from_dense([1.0, 0.0, 0.0001, 2.0], 2, 2, v => {
    Double::abs(v) > 1.0e-6
  })
  inspect(a.nnz(), content="3")
  debug_inspect(a.to_dense(), content="[1, 0, 0.0001, 2]")
}

///|
test "from_dense rejects a mis-sized array" {
  let raised = try {
    let _ = @ops.from_dense([1.0, 2.0], 3, 3, _ => true)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "drop_zeros removes stored zeros and fixes the pointers" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 2, 3], [0, 1, 1], [
    0.0, 2.0, 0.0,
  ])
  inspect(a.nnz(), content="3")
  let b = @ops.drop_zeros(a)
  inspect(b.nnz(), content="1")
  debug_inspect(b.row_ptr(), content="[0, 1, 1]")
  debug_inspect(b.col_idx(), content="[1]")
}

///|
test "scale is the same as the Csr method" {
  let a : @csr.Csr[Double] = @csr.Csr::new(1, 2, [0, 2], [0, 1], [1.0, 2.0])
  debug_inspect(@ops.scale(a, 2.0).vals(), content="[2, 4]")
}

///|
test "transpose turns a row vector into a column vector" {
  let row : @csr.Csr[Double] = @csr.Csr::new(1, 3, [0, 3], [0, 1, 2], [
    1.0, 2.0, 3.0,
  ])
  let col = @ops.transpose(row)
  inspect(col.nrows(), content="3")
  inspect(col.ncols(), content="1")
  debug_inspect(col.to_dense(), content="[1, 2, 3]")
}

///|
test "spgemm agrees with spmm on a single column" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 2, 3], [0, 1, 1], [
    1.0, 2.0, 3.0,
  ])
  let b : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [1, 0], [1.0, 4.0])
  let x = [1.0, 1.0]
  debug_inspect(@ops.spgemm(a, b).spmv(x), content="[9, 12]")
  debug_inspect(@ops.spmm(a, b.to_dense(), 2), content="[8, 1, 12, 0]")
}

///|
test "transpose and the CSC transposed product agree" {
  // Two independent routes to A^T x: build A^T as a matrix and multiply, or
  // ask CSC to apply the transpose without materialising it. They must match.
  let a : @csr.Csr[Double] = @csr.Csr::new(
    3,
    4,
    [0, 2, 3, 5],
    [0, 3, 1, 0, 2],
    [1.0, 2.0, 3.0, 4.0, 5.0],
  )
  let x = [0.5, -1.0, 2.0]
  let via_matrix = @ops.transpose(a).spmv(x)
  let via_csc = @csc.Csc::from_csr(a).spmv_transpose(x)
  debug_inspect(via_matrix, content="[8.5, -3, 10, 1]")
  debug_inspect(via_csc, content="[8.5, -3, 10, 1]")
}

///|
test "transpose of a non-symmetric rectangular matrix, entry by entry" {
  // A = [1, 0, 2 ; 3, 4, 0]   A^T = [1, 3 ; 0, 4 ; 2, 0]
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 4], [0, 2, 0, 1], [
    1.0, 2.0, 3.0, 4.0,
  ])
  let t = @ops.transpose(a)
  inspect(t.get_or(0, 0, -1.0), content="1")
  inspect(t.get_or(0, 1, -1.0), content="3")
  // (1,0) and (2,1) are zero, so they are structurally absent rather than
  // stored: get_or reports the fallback and contains reports false.
  inspect(t.contains(1, 0), content="false")
  inspect(t.get_or(1, 1, -1.0), content="4")
  inspect(t.get_or(2, 0, -1.0), content="2")
  inspect(t.contains(2, 1), content="false")
}
