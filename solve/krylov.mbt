// Krylov subspace solvers.
//
// All three start from `x = 0` and iterate until the relative residual
// `||b - A x|| / ||b||` drops below `options.tol` or the iteration budget runs
// out. None of them form a factorisation of `A`: each iteration costs one
// sparse product, which is `O(nnz)`, so a system with millions of unknowns can
// be solved in memory that a dense factorisation could not even allocate.
//
// Choosing between them:
//
//   cg        symmetric positive definite only. Cheapest per iteration
//             (one product) and needs the least storage. Raises rather than
//             silently misbehaving when handed a matrix that is not SPD.
//   bicgstab  general square matrices. Two products per iteration, no
//             transpose product, so it works on any CSR matrix.
//   gmres     general square matrices, and the most robust of the three when
//             the matrix is far from normal. Costs one product per inner
//             iteration plus storage for the whole Krylov basis, which is why
//             it restarts.

///|
/// Conjugate gradient, for symmetric positive definite systems.
///
/// The method assumes `A` is symmetric and positive definite. It cannot verify
/// that cheaply, but it does check the quantity that would go wrong: if
/// `p^T A p` is not positive the iteration is not valid for CG, and that is
/// reported as `UnsupportedMatrix` rather than allowed to produce nonsense.
pub fn cg(
  a : @csr.Csr[Double],
  b : Array[Double],
  options? : SolverOptions = SolverOptions::defaults(),
  precond? : Precond = Precond::Identity,
) -> SolveResult raise @err.SparseError {
  check_system(a, b)
  let n = a.nrows()
  let x : Array[Double] = Array::make(n, 0.0)
  let nb = norm2(b)
  if nb == 0.0 {
    // A x = 0 with x = 0 is exact.
    return { x, iterations: 0, residual: 0.0, converged: true, }
  }
  let r = b.copy()
  let mut z = precond.apply(r)
  let p = z.copy()
  let mut rz = dot(r, z)
  for iter = 1; iter <= options.max_iter; iter = iter + 1 {
    let ap = a.spmv(p)
    let pap = dot(p, ap)
    if !(pap > 0.0) {
      raise @err.SparseError::UnsupportedMatrix(
        "cg: p^T A p = " +
        pap.to_string() +
        " at iteration " +
        iter.to_string() +
        "; conjugate gradient requires a symmetric positive definite matrix",
      )
    }
    let alpha = rz / pap
    axpy(alpha, p, x)
    axpy(-alpha, ap, r)
    let res = norm2(r) / nb
    if res <= options.tol {
      return { x, iterations: iter, residual: res, converged: true, }
    }
    z = precond.apply(r)
    let rz_new = dot(r, z)
    if rz == 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "cg: the iteration broke down at step " +
        iter.to_string() +
        " (r^T M^-1 r became zero)",
      )
    }
    let beta = rz_new / rz
    // p = z + beta * p
    for i = 0; i < n; i = i + 1 {
      p[i] = z[i] + beta * p[i]
    }
    rz = rz_new
  }
  {
    x,
    iterations: options.max_iter,
    residual: norm2(r) / nb,
    converged: false,
  }
}

///|
/// BiCGSTAB, for general square systems.
///
/// Combines the bi-conjugate-gradient recurrence with a stabilising step, so it
/// handles non-symmetric matrices without the transpose product that BiCG
/// would need. Two sparse products per iteration.
pub fn bicgstab(
  a : @csr.Csr[Double],
  b : Array[Double],
  options? : SolverOptions = SolverOptions::defaults(),
  precond? : Precond = Precond::Identity,
) -> SolveResult raise @err.SparseError {
  check_system(a, b)
  let n = a.nrows()
  let x : Array[Double] = Array::make(n, 0.0)
  let nb = norm2(b)
  if nb == 0.0 {
    return { x, iterations: 0, residual: 0.0, converged: true, }
  }
  let r = b.copy()
  let r_hat = b.copy()
  let mut p : Array[Double] = Array::make(n, 0.0)
  let mut v : Array[Double] = Array::make(n, 0.0)
  let mut rho_old = 1.0
  let mut alpha = 1.0
  let mut omega = 1.0
  for iter = 1; iter <= options.max_iter; iter = iter + 1 {
    let rho = dot(r_hat, r)
    if rho == 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: the iteration broke down at step " +
        iter.to_string() +
        " (rho became zero)",
      )
    }
    if iter == 1 {
      p = r.copy()
    } else {
      if omega == 0.0 {
        raise @err.SparseError::UnsupportedMatrix(
          "bicgstab: the iteration broke down at step " +
          iter.to_string() +
          " (omega became zero)",
        )
      }
      let beta = rho / rho_old * (alpha / omega)
      // p = r + beta * (p - omega * v)
      for i = 0; i < n; i = i + 1 {
        p[i] = r[i] + beta * (p[i] - omega * v[i])
      }
    }
    let p_hat = precond.apply(p)
    v = a.spmv(p_hat)
    let rhv = dot(r_hat, v)
    if rhv == 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: the iteration broke down at step " +
        iter.to_string() +
        " (r_hat^T v became zero)",
      )
    }
    alpha = rho / rhv
    // s = r - alpha * v
    let s = sub_vec(r, scaled(alpha, v))
    let s_norm = norm2(s) / nb
    if s_norm <= options.tol {
      axpy(alpha, p_hat, x)
      return { x, iterations: iter, residual: s_norm, converged: true, }
    }
    let s_hat = precond.apply(s)
    let t = a.spmv(s_hat)
    let tt = dot(t, t)
    if tt == 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: the iteration broke down at step " +
        iter.to_string() +
        " (t became zero)",
      )
    }
    omega = dot(t, s) / tt
    // x += alpha * p_hat + omega * s_hat
    for i = 0; i < n; i = i + 1 {
      x[i] = x[i] + alpha * p_hat[i] + omega * s_hat[i]
    }
    // r = s - omega * t
    for i = 0; i < n; i = i + 1 {
      r[i] = s[i] - omega * t[i]
    }
    let res = norm2(r) / nb
    if res <= options.tol {
      return { x, iterations: iter, residual: res, converged: true, }
    }
    rho_old = rho
  }
  {
    x,
    iterations: options.max_iter,
    residual: norm2(r) / nb,
    converged: false,
  }
}

///|
/// Restarted GMRES, for general square systems.
///
/// Builds an orthonormal basis of the Krylov subspace and minimises the
/// residual over it, restarting every `restart` iterations to bound the storage
/// at `restart` vectors. The convergence test uses the residual estimated from
/// the Givens rotations, which for right preconditioning is the true residual
/// of the original system — so no extra product is needed to check it.
///
/// Preconditioning is applied on the right, which is why the estimate stays
/// meaningful: the method solves `A M^-1 u = b` and returns `x = M^-1 u`.
pub fn gmres(
  a : @csr.Csr[Double],
  b : Array[Double],
  options? : SolverOptions = SolverOptions::defaults(),
  precond? : Precond = Precond::Identity,
  restart? : Int = 30,
) -> SolveResult raise @err.SparseError {
  check_system(a, b)
  if restart <= 0 {
    raise @err.SparseError::DimensionMismatch(
      "gmres: restart must be positive, got " + restart.to_string(),
    )
  }
  let n = a.nrows()
  let x : Array[Double] = Array::make(n, 0.0)
  let nb = norm2(b)
  if nb == 0.0 {
    return { x, iterations: 0, residual: 0.0, converged: true, }
  }
  let m = restart
  // Basis vectors, stored as m+1 dense vectors of length n.
  let basis : Array[Array[Double]] = Array::make(m + 1, [])
  for j = 0; j <= m; j = j + 1 {
    basis[j] = Array::make(n, 0.0)
  }
  // Hessenberg matrix in column-major layout: h[i + j*(m+1)] is H[i][j].
  let h : Array[Double] = Array::make((m + 1) * m, 0.0)
  // Givens rotations.
  let cs : Array[Double] = Array::make(m, 0.0)
  let sn : Array[Double] = Array::make(m, 0.0)
  let g : Array[Double] = Array::make(m + 1, 0.0)
  let mut iterations = 0
  let mut residual = 1.0
  while iterations < options.max_iter {
    // r = b - A x, the true residual at the start of each restart cycle.
    let r = sub_vec(b, a.spmv(x))
    let beta = norm2(r)
    residual = beta / nb
    if residual <= options.tol {
      return { x, iterations, residual, converged: true, }
    }
    for i = 0; i <= m; i = i + 1 {
      g[i] = 0.0
    }
    g[0] = beta
    for i = 0; i < n; i = i + 1 {
      basis[0][i] = r[i] / beta
    }
    for i = 0; i < (m + 1) * m; i = i + 1 {
      h[i] = 0.0
    }
    let mut j = 0
    let mut inner_converged = false
    while j < m && iterations < options.max_iter {
      iterations = iterations + 1
      // Arnoldi step: w = A M^-1 v_j, orthogonalised against the basis so far.
      let vj_hat = precond.apply(basis[j])
      let w = a.spmv(vj_hat)
      for i = 0; i <= j; i = i + 1 {
        let hij = dot(w, basis[i])
        h[i + j * (m + 1)] = hij
        axpy(-hij, basis[i], w)
      }
      let hnext = norm2(w)
      h[j + 1 + j * (m + 1)] = hnext
      // Apply the rotations from earlier columns to this column.
      for i = 0; i < j; i = i + 1 {
        let t = cs[i] * h[i + j * (m + 1)] + sn[i] * h[i + 1 + j * (m + 1)]
        h[i + 1 + j * (m + 1)] = -sn[i] * h[i + j * (m + 1)] +
          cs[i] * h[i + 1 + j * (m + 1)]
        h[i + j * (m + 1)] = t
      }
      // New rotation that annihilates the sub-diagonal entry.
      let denom = Double::sqrt(
        h[j + j * (m + 1)] * h[j + j * (m + 1)] + hnext * hnext,
      )
      if denom == 0.0 {
        // The Krylov space closed up; the current iterate is already the
        // minimiser over it, so stop the inner loop and restart.
        inner_converged = true
      } else {
        cs[j] = h[j + j * (m + 1)] / denom
        sn[j] = hnext / denom
        h[j + j * (m + 1)] = denom
        h[j + 1 + j * (m + 1)] = 0.0
        let gnext = -sn[j] * g[j]
        g[j] = cs[j] * g[j]
        g[j + 1] = gnext
        residual = Double::abs(g[j + 1]) / nb
        if hnext != 0.0 {
          for i = 0; i < n; i = i + 1 {
            basis[j + 1][i] = w[i] / hnext
          }
        }
        if residual <= options.tol {
          // Solve the triangular system and apply the right preconditioner.
          apply_gmres_update(x, basis, h, g, m, j + 1, precond)
          return { x, iterations, residual, converged: true, }
        }
        if hnext == 0.0 {
          inner_converged = true
        }
      }
      if inner_converged {
        break
      }
      j = j + 1
    }
    // Restart: form the best iterate from the basis built in this cycle.
    if j > 0 || inner_converged {
      let used = if inner_converged && j == 0 { 0 } else { j }
      if used > 0 {
        apply_gmres_update(x, basis, h, g, m, used, precond)
      }
    }
    let after = sub_vec(b, a.spmv(x))
    residual = norm2(after) / nb
    if residual <= options.tol {
      return { x, iterations, residual, converged: true, }
    }
    if iterations >= options.max_iter {
      break
    }
  }
  { x, iterations, residual, converged: false, }
}

// Solve the upper triangular least-squares problem built during one GMRES
// cycle and fold the result into `x`. `used` is the number of columns actually
// filled in this cycle. The basis vectors live in the preconditioned variable,
// so the update is preconditioned on the way back.

///|
fn apply_gmres_update(
  x : Array[Double],
  basis : Array[Array[Double]],
  h : Array[Double],
  g : Array[Double],
  m : Int,
  used : Int,
  precond : Precond,
) -> Unit raise @err.SparseError {
  let n = x.length()
  // Back substitution on the rotated Hessenberg matrix.
  let y : Array[Double] = Array::make(used, 0.0)
  for ii = 0; ii < used; ii = ii + 1 {
    let i = used - 1 - ii
    let mut acc = g[i]
    for k = i + 1; k < used; k = k + 1 {
      acc = acc - h[i + k * (m + 1)] * y[k]
    }
    let pivot = h[i + i * (m + 1)]
    if pivot == 0.0 {
      raise @err.SparseError::SingularPivot(i, pivot)
    }
    y[i] = acc / pivot
  }
  // u = sum_j y_j * v_j, then x += M^-1 u.
  let u : Array[Double] = Array::make(n, 0.0)
  for j = 0; j < used; j = j + 1 {
    let yj = y[j]
    for i = 0; i < n; i = i + 1 {
      u[i] = u[i] + yj * basis[j][i]
    }
  }
  let correction = precond.apply(u)
  for i = 0; i < n; i = i + 1 {
    x[i] = x[i] + correction[i]
  }
}
