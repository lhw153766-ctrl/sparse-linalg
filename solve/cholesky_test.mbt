// Black-box tests for the sparse Cholesky factorisation.
//
// The central check is that the factor multiplies back to the matrix it came
// from. An index bug in the elimination tree or the reach computation produces
// a factor that is still lower triangular and still has plausible numbers in
// it, so only reconstructing L L^T catches it.

///|
fn chol_close(a : Double, b : Double, tol : Double) -> Bool {
  Double::abs(a - b) <= tol
}

///|
fn chol_spd2() -> @csr.Csr[Double] raise @err.SparseError {
  @csr.Csr::new(2, 2, [0, 2, 4], [0, 1, 0, 1], [4.0, 1.0, 1.0, 3.0])
}

// Tridiagonal Laplacian, whose factor has no fill-in.

///|
fn chol_tridiagonal(n : Int) -> @csr.Csr[Double] {
  let rp : Array[Int] = Array::make(n + 1, 0)
  let ci : Array[Int] = Array::make(3 * n - 2, 0)
  let va : Array[Double] = Array::make(3 * n - 2, 0.0)
  let mut k = 0
  for i = 0; i < n; i = i + 1 {
    if i > 0 {
      ci[k] = i - 1
      va[k] = -1.0
      k = k + 1
    }
    ci[k] = i
    va[k] = 2.0
    k = k + 1
    if i + 1 < n {
      ci[k] = i + 1
      va[k] = -1.0
      k = k + 1
    }
    rp[i + 1] = k
  }
  @csr.Csr::new_unchecked(n, n, rp, ci, va)
}

// The 5-point stencil on a 2x2 grid: SPD, and eliminating the first unknown
// creates fill-in at (1,2) and (2,1), which were zero in A.

///|
fn grid2x2() -> @csr.Csr[Double] raise @err.SparseError {
  @csr.Csr::new(4, 4, [0, 3, 6, 9, 12], [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], [
    4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0,
  ])
}

///|
test "factor of a 2x2 spd matrix is correct by hand" {
  // L = [[2, 0], [0.5, sqrt(2.75)]]
  let f = @solve.CholeskyFactor::new(chol_spd2())
  inspect(f.n(), content="2")
  inspect(chol_close(f.diag(0), 2.0, 1.0e-15), content="true")
  inspect(chol_close(f.diag(1), Double::sqrt(2.75), 1.0e-15), content="true")
  debug_inspect(f.col_nnz(0), content="2")
  debug_inspect(f.col_nnz(1), content="1")
}

///|
test "L L^T reconstructs the original matrix" {
  let a = chol_spd2()
  let f = @solve.CholeskyFactor::new(a)
  debug_inspect(f.reconstruct(), content="[4, 1, 1, 3]")
}

///|
test "L L^T reconstructs a tridiagonal matrix" {
  let n = 6
  let a = chol_tridiagonal(n)
  let f = @solve.CholeskyFactor::new(a)
  let rebuilt = f.reconstruct()
  let original = a.to_dense()
  for i = 0; i < n * n; i = i + 1 {
    inspect(chol_close(rebuilt[i], original[i], 1.0e-13), content="true")
  }
}

///|
test "L L^T reconstructs a matrix that produces fill-in" {
  let a = grid2x2()
  let f = @solve.CholeskyFactor::new(a)
  let rebuilt = f.reconstruct()
  let original = a.to_dense()
  for i = 0; i < 16; i = i + 1 {
    inspect(chol_close(rebuilt[i], original[i], 1.0e-13), content="true")
  }
}

///|
test "fill-in is real and is counted against the right baseline" {
  // The grid stencil stores both triangles: 4 diagonal entries and 8
  // off-diagonal ones. L holds only the lower triangle, so the fair comparison
  // is against 4 + 4 = 8, not against the 12 stored entries of A.
  let a = grid2x2()
  let f = @solve.CholeskyFactor::new(a)
  inspect(a.nnz(), content="12")
  let lower_triangle = 8
  // One extra entry appears: eliminating unknown 0 makes L(2,1) non-zero,
  // although A(2,1) was zero.
  inspect(f.nnz(), content="9")
  inspect(f.nnz() > lower_triangle, content="true")
  debug_inspect(f.col_ptr(), content="[0, 3, 6, 8, 9]")
  debug_inspect(f.row_idx(), content="[0, 1, 2, 1, 2, 3, 2, 3, 3]")
}

///|
test "a tridiagonal factor has no fill-in" {
  let n = 8
  let a = chol_tridiagonal(n)
  let f = @solve.CholeskyFactor::new(a)
  // n diagonal entries plus n-1 subdiagonal entries.
  inspect(f.nnz(), content="15")
  inspect(a.nnz(), content="22")
}

///|
test "solving a 2x2 system matches the hand solution" {
  let a = chol_spd2()
  let x = @solve.solve_cholesky(a, [1.0, 2.0])
  inspect(chol_close(x[0], 1.0 / 11.0, 1.0e-15), content="true")
  inspect(chol_close(x[1], 7.0 / 11.0, 1.0e-15), content="true")
}

///|
test "solving a 3x3 tridiagonal system matches the hand solution" {
  // [2,-1,0; -1,2,-1; 0,-1,2] x = [1,2,3]  ->  x = [2.5, 4, 3.5]
  let a = chol_tridiagonal(3)
  let x = @solve.solve_cholesky(a, [1.0, 2.0, 3.0])
  inspect(chol_close(x[0], 2.5, 1.0e-14), content="true")
  inspect(chol_close(x[1], 4.0, 1.0e-14), content="true")
  inspect(chol_close(x[2], 3.5, 1.0e-14), content="true")
}

///|
test "the factor solves a larger system to machine precision" {
  let n = 50
  let a = chol_tridiagonal(n)
  let b : Array[Double] = Array::make(n, 1.0)
  let f = @solve.CholeskyFactor::new(a)
  let x = f.solve(b)
  inspect(@solve.relative_residual(a, x, b) <= 1.0e-13, content="true")
}

///|
test "solving twice with one factorisation gives the same answer" {
  let n = 20
  let a = chol_tridiagonal(n)
  let f = @solve.CholeskyFactor::new(a)
  let b1 : Array[Double] = Array::make(n, 1.0)
  let b2 : Array[Double] = Array::make(n, 1.0)
  b2[0] = 5.0
  let x1 = f.solve(b1)
  let x2 = f.solve(b2)
  inspect(@solve.relative_residual(a, x1, b1) <= 1.0e-14, content="true")
  inspect(@solve.relative_residual(a, x2, b2) <= 1.0e-14, content="true")
  // The two solutions must differ, or the second solve ignored its input.
  inspect(chol_close(x1[0], x2[0], 1.0e-12), content="false")
}

///|
test "cholesky agrees with cg on the same system" {
  let n = 30
  let a = chol_tridiagonal(n)
  let b : Array[Double] = Array::make(n, 1.0)
  let direct = @solve.solve_cholesky(a, b)
  let iterative = @solve.cg(a, b)
  inspect(iterative.converged, content="true")
  for i = 0; i < n; i = i + 1 {
    inspect(chol_close(direct[i], iterative.x[i], 1.0e-9), content="true")
  }
}

///|
test "log determinant of a 2x2 matrix" {
  // det([[4,1],[1,3]]) = 11
  let f = @solve.CholeskyFactor::new(chol_spd2())
  inspect(
    chol_close(f.log_determinant(), 2.3978952727983707, 1.0e-14),
    content="true",
  )
}

///|
test "log determinant of the identity is zero" {
  let a : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 1, 2, 3], [0, 1, 2], [
    1.0, 1.0, 1.0,
  ])
  let f = @solve.CholeskyFactor::new(a)
  inspect(chol_close(f.log_determinant(), 0.0, 1.0e-15), content="true")
  debug_inspect(f.reconstruct(), content="[1, 0, 0, 0, 1, 0, 0, 0, 1]")
}

///|
test "an indefinite matrix is rejected" {
  // [[1,0],[0,-1]] is symmetric but not positive definite.
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [1.0, -1.0])
  let raised = try {
    let _ = @solve.CholeskyFactor::new(a)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "a singular matrix is rejected" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [0.0, 1.0])
  let raised = try {
    let _ = @solve.CholeskyFactor::new(a)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "a non-square matrix is rejected" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 1, 2], [0, 1], [1.0, 1.0])
  let raised = try {
    let _ = @solve.CholeskyFactor::new(a)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "an empty matrix factorises and solves trivially" {
  let a : @csr.Csr[Double] = @csr.Csr::new(0, 0, [0], [], [])
  let f = @solve.CholeskyFactor::new(a)
  inspect(f.n(), content="0")
  inspect(f.nnz(), content="0")
  debug_inspect(f.solve([]), content="[]")
}

///|
test "solve rejects a mis-sized right-hand side" {
  let f = @solve.CholeskyFactor::new(chol_spd2())
  let raised = try {
    let _ = f.solve([1.0, 2.0, 3.0])
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "factorising a matrix whose lower triangle is absent still works" {
  // Only the upper triangle is read, so a caller holding the strict upper part
  // plus the diagonal gets the same factor as one holding the full matrix.
  let upper : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 2, 3], [0, 1, 1], [
    4.0, 1.0, 3.0,
  ])
  let f = @solve.CholeskyFactor::new(upper)
  debug_inspect(f.reconstruct(), content="[4, 1, 1, 3]")
}

///|
test "the factor is lower triangular with a positive diagonal" {
  let a = grid2x2()
  let f = @solve.CholeskyFactor::new(a)
  let cp = f.col_ptr()
  let ri = f.row_idx()
  let va = f.vals()
  for j = 0; j < f.n(); j = j + 1 {
    // The diagonal is stored first, at row j, and must be positive.
    inspect(ri[cp[j]], content=j.to_string())
    inspect(va[cp[j]] > 0.0, content="true")
  }
  // Every off-diagonal entry of column j must be below the diagonal.
  for j = 0; j < f.n(); j = j + 1 {
    for p = cp[j] + 1; p < cp[j] + f.col_nnz(j); p = p + 1 {
      inspect(ri[p] > j, content="true")
    }
  }
}

///|
test "col_nnz and diag reject out-of-range columns" {
  let f = @solve.CholeskyFactor::new(chol_spd2())
  let bad_col = try {
    let _ = f.col_nnz(2)
    false
  } catch {
    _ => true
  }
  inspect(bad_col, content="true")
  let bad_diag = try {
    let _ = f.diag(-1)
    false
  } catch {
    _ => true
  }
  inspect(bad_diag, content="true")
}

///|
test "the 2x2 grid stencil factor solves its own system" {
  let a = grid2x2()
  let b = [1.0, 0.0, 0.0, 1.0]
  let x = @solve.solve_cholesky(a, b)
  inspect(@solve.relative_residual(a, x, b) <= 1.0e-14, content="true")
}
