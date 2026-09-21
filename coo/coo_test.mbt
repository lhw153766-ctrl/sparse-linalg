// Black-box tests for the COO package, including the duplicate-summing
// behaviour that makes triplet assembly safe.

///|
test "empty and push" {
  let a : @coo.Coo[Double] = @coo.Coo::empty(2, 3)
  inspect(a.nnz(), content="0")
  inspect(a.nrows(), content="2")
  inspect(a.ncols(), content="3")
  a.push(0, 1, 5.0)
  a.push(1, 2, 7.0)
  inspect(a.nnz(), content="2")
  debug_inspect(a.rows(), content="[0, 1]")
  debug_inspect(a.cols(), content="[1, 2]")
  debug_inspect(a.vals(), content="[5, 7]")
}

///|
test "push rejects out-of-range coordinates" {
  let a : @coo.Coo[Double] = @coo.Coo::empty(2, 2)
  let r = try {
    a.push(2, 0, 1.0)
    false
  } catch {
    _ => true
  }
  inspect(r, content="true")
}

///|
test "new validates lengths and bounds" {
  let bad_len = try {
    let _ : @coo.Coo[Double] = @coo.Coo::new(2, 2, [0, 1], [0], [1.0, 2.0])
    false
  } catch {
    _ => true
  }
  inspect(bad_len, content="true")
  let bad_idx = try {
    let _ : @coo.Coo[Double] = @coo.Coo::new(2, 2, [0], [5], [1.0])
    false
  } catch {
    _ => true
  }
  inspect(bad_idx, content="true")
}

///|
test "from_triplets" {
  let a : @coo.Coo[Double] = @coo.Coo::from_triplets(2, 2, [
    { row: 0, col: 0, val: 1.0, },
    { row: 1, col: 1, val: 2.0, },
  ])
  inspect(a.nnz(), content="2")
  debug_inspect(a.to_dense(), content="[1, 0, 0, 2]")
}

///|
test "duplicate coordinates are summed by to_csr" {
  // Two writes to (0,0) and one to (0,1); the (0,0) pair must add up.
  let a : @coo.Coo[Double] = @coo.Coo::new(1, 2, [0, 0, 0], [0, 0, 1], [
    1.5, 2.5, 4.0,
  ])
  inspect(a.nnz(), content="3")
  inspect(a.has_duplicates(), content="true")
  inspect(a.duplicate_count(), content="1")
  let csr = a.to_csr()
  inspect(csr.nnz(), content="2")
  inspect(csr.get_or(0, 0, 0.0), content="4")
  inspect(csr.get_or(0, 1, 0.0), content="4")
}

///|
test "to_csr sorts columns within each row" {
  // Deliberately out of order, including a duplicate pair that must merge.
  let a : @coo.Coo[Double] = @coo.Coo::new(
    2,
    4,
    [1, 0, 1, 0, 0],
    [3, 2, 1, 0, 2],
    [1.0, 2.0, 3.0, 4.0, 5.0],
  )
  let csr = a.to_csr()
  // Row 0: (0,0)=4, (0,2)=2+5=7 ; Row 1: (1,1)=3, (1,3)=1
  debug_inspect(csr.col_idx(), content="[0, 2, 1, 3]")
  debug_inspect(csr.vals(), content="[4, 7, 3, 1]")
  debug_inspect(csr.row_ptr(), content="[0, 2, 4]")
  debug_inspect(csr.to_dense(), content="[4, 0, 7, 0, 0, 3, 0, 1]")
}

///|
test "to_csr drops nothing when a duplicate pair cancels" {
  // +2 and -2 at the same coordinate leave an explicit stored zero, which is
  // the documented behaviour: structure is preserved, only values change.
  let a : @coo.Coo[Double] = @coo.Coo::new(1, 1, [0, 0], [0, 0], [2.0, -2.0])
  let csr = a.to_csr()
  inspect(csr.nnz(), content="1")
  inspect(csr.get_or(0, 0, 9.0), content="0")
}

///|
test "to_csr on an all-empty matrix" {
  let a : @coo.Coo[Double] = @coo.Coo::empty(3, 3)
  let csr = a.to_csr()
  inspect(csr.nnz(), content="0")
  debug_inspect(csr.row_ptr(), content="[0, 0, 0, 0]")
}

///|
test "get sums duplicates" {
  let a : @coo.Coo[Double] = @coo.Coo::new(1, 1, [0, 0], [0, 0], [1.0, 2.0])
  debug_inspect(a.get(0, 0), content="Some(3)")
  debug_inspect(a.get(0, 0), content="Some(3)")
  let b : @coo.Coo[Double] = @coo.Coo::empty(1, 1)
  debug_inspect(b.get(0, 0), content="None")
}

///|
test "transpose swaps the shape" {
  let a : @coo.Coo[Double] = @coo.Coo::new(2, 3, [0, 1], [2, 0], [5.0, 6.0])
  let t = a.transpose()
  inspect(t.nrows(), content="3")
  inspect(t.ncols(), content="2")
  debug_inspect(t.rows(), content="[2, 0]")
  debug_inspect(t.cols(), content="[0, 1]")
  debug_inspect(t.to_dense(), content="[0, 6, 0, 0, 5, 0]")
}

///|
test "scale and sum_vals" {
  let a : @coo.Coo[Double] = @coo.Coo::new(1, 2, [0, 0], [0, 1], [2.0, 3.0])
  inspect(a.sum_vals(), content="5")
  debug_inspect(a.scale(2.0).vals(), content="[4, 6]")
  debug_inspect(a.vals(), content="[2, 3]")
  let e : @coo.Coo[Double] = @coo.Coo::empty(1, 1)
  inspect(e.scale(2.0).nnz(), content="0")
}

///|
test "no duplicates reported for a clean matrix" {
  let a : @coo.Coo[Double] = @coo.Coo::new(2, 2, [0, 1], [0, 1], [1.0, 2.0])
  inspect(a.has_duplicates(), content="false")
  inspect(a.duplicate_count(), content="0")
}

///|
test "from_csr round trips" {
  let csr : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 2, 3], [0, 2, 1], [
    1.0, 2.0, 3.0,
  ])
  let coo = @coo.Coo::from_csr(csr)
  inspect(coo.nnz(), content="3")
  debug_inspect(coo.rows(), content="[0, 0, 1]")
  debug_inspect(coo.cols(), content="[0, 2, 1]")
  let back = coo.to_csr()
  debug_inspect(back.col_idx(), content="[0, 2, 1]")
  debug_inspect(back.vals(), content="[1, 2, 3]")
}

///|
test "Int64 values stay exact through assembly" {
  let a : @coo.Coo[Int64] = @coo.Coo::new(1, 1, [0, 0], [0, 0], [
    9000000000000L, 7L,
  ])
  let csr = a.to_csr()
  inspect(csr.get_or(0, 0, 0L), content="9000000000007")
}
