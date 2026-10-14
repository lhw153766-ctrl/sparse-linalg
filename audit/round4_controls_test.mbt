///|
test "r4 GMRES consistent singular and negative eigensystem controls" {
  let a = @ops.from_dense([1.0, 0.0, 0.0, 0.0], 2, 2, x => x != 0.0)
  let r = @solve.gmres(a, [1.0, 0.0])
  assert_true(r.converged)
  assert_eq(r.x, [1.0, 0.0])
  let b = @ops.identity(1, -1.0)
  assert_true(
    r4_rejected(() => {
      let _ = @solve.cg(b, [1.0])
    }),
  )
  assert_true(@solve.gmres(b, [1.0]).converged)
  assert_true(@solve.bicgstab(b, [1.0]).converged)
}

///|
test "r4 invalid public options and restart capacity rejected" {
  let a = @ops.identity(1, 1.0)
  let opts : @solve.SolverOptions = { tol: @double.infinity, max_iter: 1, }
  assert_true(
    r4_rejected(() => {
      let _ = @solve.cg(a, [1.0], options=opts)
    }),
  )
  assert_true(
    r4_rejected(() => {
      let _ = @solve.bicgstab(a, [1.0], options=opts)
    }),
  )
  assert_true(
    r4_rejected(() => {
      let _ = @solve.gmres(a, [1.0], options=opts)
    }),
  )
  assert_true(
    r4_rejected(() => {
      let _ = @solve.gmres(a, [1.0], restart=46341)
    }),
  )
}

///|
test "r4 MM header and size overflow boundary rejected" {
  assert_true(
    r4_rejected(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix array real general\n65536 65536\n",
      )
    }),
  )
  assert_true(
    r4_rejected(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n-2147483648 1 2\n",
      )
    }),
  )
  let a = @io.parse_matrix_market(
    "%%MatrixMarket matrix coordinate real general\n0 0 0\n%tail\n",
  )
  assert_eq(a.nnz(), 0)
}
