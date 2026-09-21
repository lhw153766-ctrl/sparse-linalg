// Black-box tests for the solvers.
//
// Small systems are used deliberately: they can be solved by hand, so the
// expected solution is derived on paper rather than taken from the solver's own
// output. The randomised differential tests against a dense reference live in
// the module-level `differential` package.

// A tridiagonal matrix: the 1D Laplacian with Dirichlet boundaries, scaled.
// Its ILU(0) factorisation is exact, because eliminating a tridiagonal matrix
// introduces no fill-in. That property is used below.

///|
fn tridiagonal(n : Int) -> @csr.Csr[Double] {
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

// [[4, 1], [1, 3]] is symmetric positive definite (leading minors 4 and 11).

///|
fn spd2() -> @csr.Csr[Double] raise @err.SparseError {
  @csr.Csr::new(2, 2, [0, 2, 4], [0, 1, 0, 1], [4.0, 1.0, 1.0, 3.0])
}

// [[3, 2], [1, 4]] is neither symmetric nor definite.

///|
fn nonsym2() -> @csr.Csr[Double] raise @err.SparseError {
  @csr.Csr::new(2, 2, [0, 2, 4], [0, 1, 0, 1], [3.0, 2.0, 1.0, 4.0])
}

///|
fn close(a : Double, b : Double, tol : Double) -> Bool {
  Double::abs(a - b) <= tol
}

///|
test "options validate their inputs" {
  let o = @solve.SolverOptions::defaults()
  inspect(o.max_iter, content="1000")
  let custom = @solve.SolverOptions::new(1.0e-6, 50)
  inspect(custom.max_iter, content="50")
  let bad_tol = try {
    let _ = @solve.SolverOptions::new(0.0, 10)
    false
  } catch {
    _ => true
  }
  inspect(bad_tol, content="true")
  let bad_iter = try {
    let _ = @solve.SolverOptions::new(1.0e-6, 0)
    false
  } catch {
    _ => true
  }
  inspect(bad_iter, content="true")
}

///|
test "vector helpers" {
  inspect(@solve.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), content="32")
  inspect(@solve.norm2([3.0, 4.0]), content="5")
  inspect(@solve.norm2([]), content="0")
  let y = [1.0, 1.0]
  @solve.axpy(2.0, [1.0, 2.0], y)
  debug_inspect(y, content="[3, 5]")
  debug_inspect(@solve.scaled(3.0, [1.0, 2.0]), content="[3, 6]")
  debug_inspect(@solve.sub_vec([5.0, 5.0], [1.0, 2.0]), content="[4, 3]")
  debug_inspect(@solve.add_vec([1.0, 2.0], [3.0, 4.0]), content="[4, 6]")
  let mismatch = try {
    let _ = @solve.dot([1.0], [1.0, 2.0])
    false
  } catch {
    _ => true
  }
  inspect(mismatch, content="true")
}

///|
test "cg solves a 2x2 SPD system" {
  // [[4,1],[1,3]] x = [1,2]  ->  x = [1/11, 7/11]
  let a = spd2()
  let result = @solve.cg(a, [1.0, 2.0])
  inspect(result.converged, content="true")
  inspect(close(result.x[0], 1.0 / 11.0, 1.0e-12), content="true")
  inspect(close(result.x[1], 7.0 / 11.0, 1.0e-12), content="true")
  inspect(result.residual <= 1.0e-10, content="true")
}

///|
test "cg on a tridiagonal system reaches the tolerance" {
  let n = 40
  let a = tridiagonal(n)
  let b : Array[Double] = Array::make(n, 1.0)
  let result = @solve.cg(a, b)
  inspect(result.converged, content="true")
  inspect(result.residual <= 1.0e-10, content="true")
  // Independently confirm with the residual function rather than trusting the
  // value the solver reported.
  inspect(@solve.relative_residual(a, result.x, b) <= 1.0e-10, content="true")
}

///|
test "cg with an all-zero right-hand side converges immediately" {
  let a = spd2()
  let result = @solve.cg(a, [0.0, 0.0])
  inspect(result.converged, content="true")
  inspect(result.iterations, content="0")
  debug_inspect(result.x, content="[0, 0]")
}

///|
test "cg rejects a matrix that is not positive definite" {
  // [[1, 0], [0, -1]] is symmetric but indefinite.
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [0, 1], [1.0, -1.0])
  let raised = try {
    let _ = @solve.cg(a, [1.0, 1.0])
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "cg rejects a non-square system and a mis-sized right-hand side" {
  let rect : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 1, 2], [0, 1], [
    1.0, 1.0,
  ])
  let not_square = try {
    let _ = @solve.cg(rect, [1.0, 1.0])
    false
  } catch {
    _ => true
  }
  inspect(not_square, content="true")
  let a = spd2()
  let wrong_size = try {
    let _ = @solve.cg(a, [1.0, 2.0, 3.0])
    false
  } catch {
    _ => true
  }
  inspect(wrong_size, content="true")
}

///|
test "cg reports failure when the budget is too small" {
  let n = 60
  let a = tridiagonal(n)
  let b : Array[Double] = Array::make(n, 1.0)
  let result = @solve.cg(a, b, options=@solve.SolverOptions::new(1.0e-14, 1))
  inspect(result.converged, content="false")
  inspect(result.iterations, content="1")
}

///|
test "jacobi preconditioning cuts the iteration count" {
  // A badly scaled SPD system: the diagonal ranges over four orders of
  // magnitude, which is exactly what diagonal scaling is for.
  let a : @csr.Csr[Double] = @csr.Csr::new(
    3,
    3,
    [0, 2, 4, 6],
    [0, 1, 0, 1, 1, 2],
    [1.0e4, 1.0, 1.0, 1.0e2, 1.0, 1.0],
  )
  let b = [1.0, 1.0, 1.0]
  let opts = @solve.SolverOptions::new(1.0e-10, 500)
  let plain = @solve.cg(a, b, options=opts)
  let jac = @solve.cg(
    a,
    b,
    options=opts,
    precond=@solve.Precond::Jacobi(@solve.JacobiPrecond::new(a)),
  )
  inspect(plain.converged, content="true")
  inspect(jac.converged, content="true")
  inspect(jac.iterations < plain.iterations, content="true")
}

///|
test "jacobi preconditioning reaches a residual plain cg cannot" {
  // With the same budget and a tolerance neither solver can hit, the
  // preconditioned run still gets much closer. On this matrix plain CG stalls
  // around 2e-10, which is the double precision floor for a condition number
  // of about 1e4, while Jacobi keeps making progress.
  let a : @csr.Csr[Double] = @csr.Csr::new(
    3,
    3,
    [0, 2, 4, 6],
    [0, 1, 0, 1, 1, 2],
    [1.0e4, 1.0, 1.0, 1.0e2, 1.0, 1.0],
  )
  let b = [1.0, 1.0, 1.0]
  let too_strict = @solve.SolverOptions::new(1.0e-16, 50)
  let plain = @solve.cg(a, b, options=too_strict)
  let jac = @solve.cg(
    a,
    b,
    options=too_strict,
    precond=@solve.Precond::Jacobi(@solve.JacobiPrecond::new(a)),
  )
  inspect(plain.converged, content="false")
  inspect(jac.converged, content="false")
  inspect(jac.residual < plain.residual, content="true")
}

///|
test "jacobi preconditioner is the reciprocal diagonal" {
  let a = spd2()
  let p = @solve.JacobiPrecond::new(a)
  debug_inspect(p.inv_diag(), content="[0.25, 0.3333333333333333]")
  debug_inspect(p.apply([1.0, 1.0]), content="[0.25, 0.3333333333333333]")
}

///|
test "jacobi preconditioner rejects a zero diagonal" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 1], [0], [0.0])
  let raised = try {
    let _ = @solve.JacobiPrecond::new(a)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "ilu0 is exact on a tridiagonal matrix" {
  // Eliminating a tridiagonal matrix introduces no fill, so the zero-fill
  // factorisation is the complete one and M^-1 b is the exact solution.
  let n = 5
  let a = tridiagonal(n)
  let b = [1.0, 0.0, 0.0, 0.0, 0.0]
  let p = @solve.Ilu0Precond::new(a)
  let x = p.apply(b)
  inspect(@solve.relative_residual(a, x, b) <= 1.0e-14, content="true")
}

///|
test "ilu0 keeps the sparsity pattern of the matrix" {
  let a = tridiagonal(6)
  let p = @solve.Ilu0Precond::new(a)
  inspect(p.nnz(), content="16")
}

///|
test "ilu0 rejects a matrix with a missing diagonal entry" {
  // Row 0 has no entry in column 0.
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [1, 0], [1.0, 1.0])
  let raised = try {
    let _ = @solve.Ilu0Precond::new(a)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "ilu0 as a preconditioner speeds up cg on a tridiagonal system" {
  let n = 200
  let a = tridiagonal(n)
  let b : Array[Double] = Array::make(n, 1.0)
  let plain = @solve.cg(a, b, options=@solve.SolverOptions::new(1.0e-10, 2000))
  let pre = @solve.cg(
    a,
    b,
    options=@solve.SolverOptions::new(1.0e-10, 2000),
    precond=@solve.Precond::Ilu0(@solve.Ilu0Precond::new(a)),
  )
  inspect(plain.converged, content="true")
  inspect(pre.converged, content="true")
  // The exact factorisation means the preconditioned system is solved in a
  // single step, so this is a strict improvement, not a tie.
  inspect(pre.iterations < plain.iterations, content="true")
  inspect(pre.iterations, content="1")
}

///|
test "bicgstab solves a non-symmetric system" {
  // [[3,2],[1,4]] x = [1,1]  ->  x = [0.2, 0.2]
  let a = nonsym2()
  let result = @solve.bicgstab(a, [1.0, 1.0])
  inspect(result.converged, content="true")
  inspect(close(result.x[0], 0.2, 1.0e-12), content="true")
  inspect(close(result.x[1], 0.2, 1.0e-12), content="true")
}

///|
test "bicgstab on a non-symmetric convection-like system" {
  // A tridiagonal with an asymmetric off-diagonal pair, which is what a
  // first-order upwind discretisation produces.
  let n = 30
  let rp : Array[Int] = Array::make(n + 1, 0)
  let ci : Array[Int] = Array::make(3 * n - 2, 0)
  let va : Array[Double] = Array::make(3 * n - 2, 0.0)
  let mut k = 0
  for i = 0; i < n; i = i + 1 {
    if i > 0 {
      ci[k] = i - 1
      va[k] = -2.0
      k = k + 1
    }
    ci[k] = i
    va[k] = 4.0
    k = k + 1
    if i + 1 < n {
      ci[k] = i + 1
      va[k] = -1.0
      k = k + 1
    }
    rp[i + 1] = k
  }
  let a = @csr.Csr::new_unchecked(n, n, rp, ci, va)
  let b : Array[Double] = Array::make(n, 1.0)
  let result = @solve.bicgstab(a, b)
  inspect(result.converged, content="true")
  inspect(@solve.relative_residual(a, result.x, b) <= 1.0e-10, content="true")
}

///|
test "gmres solves a non-symmetric system" {
  let a = nonsym2()
  let result = @solve.gmres(a, [1.0, 1.0])
  inspect(result.converged, content="true")
  inspect(close(result.x[0], 0.2, 1.0e-12), content="true")
  inspect(close(result.x[1], 0.2, 1.0e-12), content="true")
}

///|
test "gmres on a larger non-symmetric system" {
  let n = 50
  let rp : Array[Int] = Array::make(n + 1, 0)
  let ci : Array[Int] = Array::make(3 * n - 2, 0)
  let va : Array[Double] = Array::make(3 * n - 2, 0.0)
  let mut k = 0
  for i = 0; i < n; i = i + 1 {
    if i > 0 {
      ci[k] = i - 1
      va[k] = -2.0
      k = k + 1
    }
    ci[k] = i
    va[k] = 4.0
    k = k + 1
    if i + 1 < n {
      ci[k] = i + 1
      va[k] = -1.0
      k = k + 1
    }
    rp[i + 1] = k
  }
  let a = @csr.Csr::new_unchecked(n, n, rp, ci, va)
  let b : Array[Double] = Array::make(n, 1.0)
  let result = @solve.gmres(a, b)
  inspect(result.converged, content="true")
  inspect(@solve.relative_residual(a, result.x, b) <= 1.0e-10, content="true")
}

///|
test "gmres with a small restart still converges on a small system" {
  // A restart shorter than the dimension of the Krylov space needed for
  // convergence costs extra cycles, so keep the system small here.
  let n = 10
  let a = tridiagonal(n)
  let b : Array[Double] = Array::make(n, 1.0)
  let result = @solve.gmres(
    a,
    b,
    options=@solve.SolverOptions::new(1.0e-10, 200),
    restart=3,
  )
  inspect(result.converged, content="true")
  inspect(@solve.relative_residual(a, result.x, b) <= 1.0e-10, content="true")
}

///|
test "a shorter gmres restart costs more iterations but agrees on the answer" {
  let n = 20
  let a = tridiagonal(n)
  let b : Array[Double] = Array::make(n, 1.0)
  let opts = @solve.SolverOptions::new(1.0e-10, 2000)
  let short = @solve.gmres(a, b, options=opts, restart=5)
  let long = @solve.gmres(a, b, options=opts, restart=30)
  inspect(short.converged, content="true")
  inspect(long.converged, content="true")
  // Restarting throws the Krylov basis away, so a shorter restart cannot need
  // fewer iterations on the same problem.
  inspect(short.iterations > long.iterations, content="true")
  for i = 0; i < n; i = i + 1 {
    inspect(close(short.x[i], long.x[i], 1.0e-8), content="true")
  }
}

///|
test "gmres rejects a non-positive restart" {
  let a = spd2()
  let raised = try {
    let _ = @solve.gmres(a, [1.0, 1.0], restart=0)
    false
  } catch {
    _ => true
  }
  inspect(raised, content="true")
}

///|
test "all three solvers agree on the same system" {
  let n = 20
  let a = tridiagonal(n)
  let b : Array[Double] = Array::make(n, 1.0)
  let cg_result = @solve.cg(a, b)
  let bi_result = @solve.bicgstab(a, b)
  let gm_result = @solve.gmres(a, b)
  for i = 0; i < n; i = i + 1 {
    inspect(close(cg_result.x[i], bi_result.x[i], 1.0e-9), content="true")
    inspect(close(cg_result.x[i], gm_result.x[i], 1.0e-9), content="true")
  }
}

///|
test "precond names and selection" {
  let a = tridiagonal(4)
  inspect(@solve.Precond::Identity.name(), content="identity")
  inspect(@solve.Precond::for_matrix("jacobi", a).name(), content="jacobi")
  inspect(@solve.Precond::for_matrix("ilu0", a).name(), content="ilu0")
  inspect(@solve.Precond::for_matrix("identity", a).name(), content="identity")
  let unknown = try {
    let _ = @solve.Precond::for_matrix("nope", a)
    false
  } catch {
    _ => true
  }
  inspect(unknown, content="true")
}

///|
test "the identity preconditioner is a pass-through" {
  debug_inspect(@solve.Precond::Identity.apply([1.0, -2.0]), content="[1, -2]")
}

///|
test "relative_residual reports a wrong solution as not small" {
  let a = spd2()
  inspect(
    @solve.relative_residual(a, [1.0 / 11.0, 7.0 / 11.0], [1.0, 2.0]) <= 1.0e-14,
    content="true",
  )
  inspect(
    @solve.relative_residual(a, [0.0, 0.0], [1.0, 2.0]) > 0.5,
    content="true",
  )
}

///|
// The 5-point stencil on a `k x k` grid, built directly as CSR arrays so the
// solver tests do not need the assembly package.
fn warm_grid(k : Int) -> @csr.Csr[Double] {
  let n = k * k
  let rp : Array[Int] = Array::make(n + 1, 0)
  let ci : Array[Int] = Array(capacity=5 * n)
  let va : Array[Double] = Array(capacity=5 * n)
  for r = 0; r < k; r = r + 1 {
    for c = 0; c < k; c = c + 1 {
      let v = r * k + c
      let cols : Array[Int] = [v]
      if r > 0 {
        cols.push(v - k)
      }
      if r + 1 < k {
        cols.push(v + k)
      }
      if c > 0 {
        cols.push(v - 1)
      }
      if c + 1 < k {
        cols.push(v + 1)
      }
      cols.sort()
      for col in cols {
        ci.push(col)
        va.push(if col == v { 4.0 } else { -1.0 })
      }
      rp[v + 1] = ci.length()
    }
  }
  @csr.Csr::new_unchecked(n, n, rp, ci, va)
}

///|
test "a warm start saves iterations on a 2D grid" {
  // Reusing the previous solution of a slowly changing system is what a warm
  // start is for, and here it pays: starting from a slightly perturbed solution
  // takes about a third fewer iterations than starting from zero.
  //
  // It does not always pay. On a 1D tridiagonal system with a constant
  // right-hand side, cold and warm both take the same number of iterations,
  // because there the convergence is limited by the smoothness of the
  // right-hand side rather than by the size of the starting residual.
  let k = 20
  let a = warm_grid(k)
  let n = k * k
  let b : Array[Double] = Array::make(n, 1.0)
  let cold = @solve.cg(a, b)
  inspect(cold.converged, content="true")
  let perturbed : Array[Double] = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    perturbed[i] = cold.x[i] * 1.000001
  }
  let warm = @solve.cg(a, b, initial=Some(perturbed))
  inspect(warm.converged, content="true")
  inspect(warm.iterations < cold.iterations, content="true")
  inspect(@solve.relative_residual(a, warm.x, b) <= 1.0e-10, content="true")
  // The answer must be the same one, reached sooner.
  for i = 0; i < n; i = i + 1 {
    inspect(Double::abs(warm.x[i] - cold.x[i]) < 1.0e-8, content="true")
  }
}

///|
test "a warm start that is already the answer returns immediately" {
  let n = 20
  let a = tridiagonal(n)
  let b : Array[Double] = Array::make(n, 1.0)
  let exact = @solve.solve_cholesky(a, b)
  let warm = @solve.cg(a, b, initial=Some(exact))
  inspect(warm.converged, content="true")
  inspect(warm.iterations, content="0")
  inspect(warm.residual <= 1.0e-10, content="true")
}

///|
test "all three solvers accept a warm start" {
  let a = nonsym2()
  let b = [1.0, 1.0]
  let exact = [0.2, 0.2]
  let via_bi = @solve.bicgstab(a, b, initial=Some(exact))
  inspect(via_bi.iterations, content="0")
  let via_gm = @solve.gmres(a, b, initial=Some(exact))
  inspect(via_gm.iterations, content="0")
  inspect(@solve.relative_residual(a, via_gm.x, b) <= 1.0e-12, content="true")
}

///|
test "a warm start of the wrong length is rejected" {
  let a = spd2()
  let too_long = try {
    let _ = @solve.cg(a, [1.0, 2.0], initial=Some([1.0, 2.0, 3.0]))
    false
  } catch {
    _ => true
  }
  inspect(too_long, content="true")
  let too_short = try {
    let _ = @solve.bicgstab(a, [1.0, 2.0], initial=Some([1.0]))
    false
  } catch {
    _ => true
  }
  inspect(too_short, content="true")
}

///|
test "the starting point is copied, not aliased" {
  // A solver must not write into the caller's array.
  let a = spd2()
  let b = [1.0, 2.0]
  let start = [0.0, 0.0]
  let _ = @solve.cg(a, b, initial=Some(start))
  debug_inspect(start, content="[0, 0]")
}
