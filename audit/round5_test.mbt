///|
test "r5 mixed scale warm starts cannot certify subnormal residual loss" {
  let u = 5.0e-324
  let a = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [1.0, u])
  let b = [0.0, u]
  let initial = [0.0, 0.75]
  assert_eq(@solve.relative_residual(a, initial, b), 0.25)
  assert_eq(@solve.relative_residual(a, [0.0, 1.0], b), 0.0)
  let options = @solve.SolverOptions::new(1.0e-10, 2)
  let solvers : Array[() -> @solve.SolveResult raise @err.SparseError] = [
    () => @solve.cg(a, b, options~, initial=Some(initial)),
    () => @solve.bicgstab(a, b, options~, initial=Some(initial)),
    () => @solve.gmres(a, b, options~, initial=Some(initial)),
  ]
  for solve in solvers {
    let result = Some(solve()) catch {
      @err.SparseError::UnsupportedMatrix(_) => None
      e => raise e
    }
    if result is Some(result) {
      assert_true(@solve.is_finite_vector(result.x))
      assert_true(!result.converged || result.x[1] == 1.0)
      assert_true(!result.converged || result.residual <= options.tol)
      if result.x[1] == 0.75 {
        assert_eq(result.residual, 0.25)
      }
    }
  }
}

///|
test "r5 residual evaluates products beyond Double range without overflow" {
  let a = @csr.Csr::new(1, 2, [0, 2], [0, 1], [1.0e308, -1.0e308])
  assert_eq(@solve.relative_residual(a, [2.0, 2.0], [1.0e308]), 1.0)
  let a = @ops.identity(1, 5.0e-324)
  assert_eq(@solve.relative_residual(a, [0.75], [5.0e-324]), 0.25)
  let a = @ops.identity(2, 1.0)
  assert_eq(@solve.relative_residual(a, [0.0, 0.0], [1.3e308, 1.3e308]), 1.0)
}

///|
test "r5 residual retains small terms during cancellation" {
  let a = @csr.Csr::new(1, 3, [0, 3], [0, 1, 2], [1.0e16, 1.0, -1.0e16])
  assert_eq(@solve.relative_residual(a, [1.0, 1.0, 1.0], [0.0]), 1.0)
  assert_eq(@solve.relative_residual(a, [1.0, 1.0, 1.0], [1.0]), 0.0)
  let extreme = @csr.Csr::new(1, 2, [0, 2], [0, 1], [1.0e308, 5.0e-324])
  let rejected = try {
    let _ = @solve.relative_residual(extreme, [1.0, 1.0], [1.0e308])
    false
  } catch {
    @err.SparseError::UnsupportedMatrix(_) => true
    _ => false
  }
  assert_true(rejected)
}

///|
test "r5 residual retains multiplication rounding errors at strict tolerances" {
  let a = @ops.identity(1, 1.0000000000000002)
  let x = [0.9999999999999998]
  // (1 + 2^-52)(1 - 2^-52) rounds to 1 in an ordinary SpMV.
  let residual = @solve.relative_residual(a, x, [1.0])
  assert_true(residual > 4.9e-32 && residual < 5.0e-32)
  let options = @solve.SolverOptions::new(1.0e-35, 2)
  let result = Some(@solve.gmres(a, [1.0], options~, initial=Some(x))) catch {
    @err.SparseError::UnsupportedMatrix(_) => None
    e => raise e
  }
  if result is Some(result) {
    assert_true(!result.converged)
    assert_true(result.residual > options.tol)
  }
}

///|
test "r5 zero dense shapes cannot bypass sparse dimension bounds" {
  for
    (rows, cols) in [
      (2147483647, 0),
      (0, 2147483647),
      (1_000_001, 0),
      (0, 1_000_001),
    ] {
    let rejected = try {
      let _ = @ops.from_dense([], rows, cols, fn(_ : Double) { true })
      false
    } catch {
      @err.SparseError::DimensionMismatch(_) => true
      _ => false
    }
    assert_true(rejected)
  }
  let zero_rows = @ops.from_dense([], 0, 2, fn(_ : Double) { true })
  let zero_cols = @ops.from_dense([], 2, 0, fn(_ : Double) { true })
  assert_eq(zero_rows.row_ptr(), [0])
  assert_eq(zero_cols.row_ptr(), [0, 0, 0])
}

///|
test "r5 GMRES rejects aggregate workspace before allocating huge arrays" {
  let a = @ops.identity(1, 1.0)
  let rejected = try {
    let _ = @solve.gmres(a, [1.0], restart=46340)
    false
  } catch {
    @err.SparseError::DimensionMismatch(_) => true
    _ => false
  }
  assert_true(rejected)
  // Each array could individually fit the limit, but the basis plus H cannot.
  let a = @ops.identity(10_000, 1.0)
  let rejected = try {
    let _ = @solve.gmres(a, Array::make(10_000, 1.0), restart=3000)
    false
  } catch {
    @err.SparseError::DimensionMismatch(_) => true
    _ => false
  }
  assert_true(rejected)
}

///|
test "r5 Matrix Market does not concatenate embedded CR or accept numeric extensions" {
  for
    value in [
      "1\r2", "1_000", "1junk", "1e+", "+", ".", "NaN", "Infinity", "-Infinity",
      "1e999",
    ] {
    let text = "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 " +
      value +
      "\n"
    let rejected = try {
      let _ = @io.parse_matrix_market(text)
      false
    } catch {
      @err.SparseError::ParseError(_) => true
      _ => false
    }
    assert_true(rejected)
  }
  let a = @io.parse_matrix_market(
    "%%MatrixMarket matrix array real symmetric\r\n %before\r\n3 3\r\n1 2 3\r\n \r\n%middle\r\n4\t5 6\r\n%tail\r\n",
  )
  assert_eq(a.to_dense(), [1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0])
  let a = @io.parse_matrix_market(
    "%%MatrixMarket matrix array real general\n1 3\n+.5 -2. 1E-2\n",
  )
  assert_eq(a.to_dense(), [0.5, -2.0, 0.01])
}

///|
test "r5 Jacobi rejects unrepresentable reciprocals and nonfinite applications" {
  for diagonal in [5.0e-324, @double.infinity, -@double.infinity, 0.0 / 0.0] {
    let a = @ops.identity(1, diagonal)
    let rejected = try {
      let _ = @solve.JacobiPrecond::new(a)
      false
    } catch {
      @err.SparseError::UnsupportedMatrix(_) => true
      _ => false
    }
    assert_true(rejected)
  }
  let a = @ops.identity(1, 1.0e-300)
  let p = @solve.JacobiPrecond::new(a)
  assert_true(Double::abs(p.apply([1.0e-300])[0] - 1.0) < 1.0e-15)
  let rejected = try {
    let _ = p.apply([1.0e300])
    false
  } catch {
    @err.SparseError::UnsupportedMatrix(_) => true
    _ => false
  }
  assert_true(rejected)
}

///|
test "r5 ILU rejects nonfinite matrices and singular completed factors" {
  for diagonal in [@double.infinity, 0.0 / 0.0, 0.0] {
    let a = @ops.identity(1, diagonal)
    let rejected = try {
      let _ = @solve.Ilu0Precond::new(a)
      false
    } catch {
      @err.SparseError::UnsupportedMatrix(_) => true
      @err.SparseError::SingularPivot(_, _) => true
      _ => false
    }
    assert_true(rejected)
  }
}

///|
test "r5 SpGEMM cancellation and explicit zeros keep unique sorted columns" {
  let a = @csr.Csr::new(1, 4, [0, 4], [0, 1, 2, 3], [0.0, 1.0, -1.0, 2.0])
  let b = @csr.Csr::new(4, 1, [0, 1, 2, 3, 4], [0, 0, 0, 0], [
    1.0, 1.0, 1.0, 1.0,
  ])
  let product = @ops.spgemm(a, b)
  assert_eq(product.row_ptr(), [0, 1])
  assert_eq(product.col_idx(), [0])
  assert_eq(product.vals(), [2.0])
}

///|
test "r5 capacity guards check widened counts before narrowing or allocation" {
  assert_eq(@err.check_sparse_nnz(10_000_000L), 10_000_000)
  @err.check_workspace_capacity(30_000_000L)
  let guards : Array[() -> Unit raise @err.SparseError] = [
    () => {
      let _ = @err.check_sparse_nnz(10_000_001L)
    },
    () => {
      let _ = @err.check_sparse_nnz(2147483648L)
    },
    () => {
      let _ = @err.check_sparse_nnz(-1L)
    },
    () => @err.check_workspace_capacity(30_000_001L),
    () => @err.check_sparse_shape(0, 2147483647),
    () => {
      let _ = @ops.identity(30_000_000, 1.0)
    },
    () => {
      let _ = @err.check_dense_capacity(65536, 65536)
    },
  ]
  for check in guards {
    let rejected = try {
      check()
      false
    } catch {
      @err.SparseError::DimensionMismatch(_) => true
      _ => false
    }
    assert_true(rejected)
  }
}
