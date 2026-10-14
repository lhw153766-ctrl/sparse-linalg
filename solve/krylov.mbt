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
fn system_magnitude(a : @csr.Csr[Double], b : Array[Double]) -> Double {
  let mut m = 0.0
  for v in b {
    let abs_v = Double::abs(v)
    if abs_v > m {
      m = abs_v
    }
  }
  let va = a.vals()
  for v in va {
    let abs_v = Double::abs(v)
    if abs_v > m {
      m = abs_v
    }
  }
  m
}

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
  initial? : Array[Double]? = None,
) -> SolveResult raise @err.SparseError {
  check_system(a, b)
  check_solver_options(options)
  let mag = system_magnitude(a, b)
  if mag < 1.0e-100 && mag > 0.0 {
    let s = SUBNORMAL_SCALE_FACTOR
    let scaled_a = a.scale(s)
    let scaled_b : Array[Double] = Array::make(b.length(), 0.0)
    for i = 0; i < b.length(); i = i + 1 {
      scaled_b[i] = b[i] * s
    }
    return cg(scaled_a, scaled_b, options~, precond~, initial~)
  }
  let n = a.nrows()
  let x = starting_point(a, initial)
  let nb = norm2(b)
  if nb.is_nan() || nb.is_inf() {
    raise @err.SparseError::UnsupportedMatrix(
      "cg: right-hand side norm is non-finite or overflows floating-point representation",
    )
  }
  if nb == 0.0 {
    // A x = 0 is solved exactly by x = 0.
    return {
      x: Array::make(n, 0.0),
      iterations: 0,
      residual: 0.0,
      converged: true,
    }
  }
  let r = sub_vec(b, a.spmv(x))
  let first_residual = norm2(r) / nb
  if first_residual <= options.tol {
    if is_finite_vector(x) {
      let true_res = relative_residual(a, x, b)
      if is_finite_val(true_res) && true_res <= options.tol {
        return { x, iterations: 0, residual: true_res, converged: true, }
      }
    }
  }
  let mut z = precond.apply(r)
  let p = z.copy()
  let mut rz = dot(r, z)
  for iter = 1; iter <= options.max_iter; iter = iter + 1 {
    if rz.is_nan() || rz.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "cg: numerical breakdown: r^T M^-1 r is non-finite at iteration " +
        iter.to_string(),
      )
    }
    if rz == 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "cg: numerical breakdown: r^T M^-1 r became zero at iteration " +
        iter.to_string() +
        " (underflow)",
      )
    }
    let ap = a.spmv(p)
    let pap = dot(p, ap)
    if pap.is_nan() || pap.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "cg: numerical breakdown: p^T A p is non-finite at iteration " +
        iter.to_string(),
      )
    }
    if pap < 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "cg: p^T A p = " +
        pap.to_string() +
        " at iteration " +
        iter.to_string() +
        "; conjugate gradient requires a symmetric positive definite matrix",
      )
    }
    if pap == 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "cg: numerical breakdown: p^T A p became zero at iteration " +
        iter.to_string() +
        " (underflow)",
      )
    }
    let alpha = rz / pap
    if alpha.is_nan() || alpha.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "cg: numerical breakdown: alpha is non-finite at iteration " +
        iter.to_string(),
      )
    }
    axpy(alpha, p, x)
    axpy(-alpha, ap, r)
    let res = norm2(r) / nb
    if res <= options.tol {
      if is_finite_vector(x) {
        let true_res = relative_residual(a, x, b)
        if is_finite_val(true_res) && true_res <= options.tol {
          return { x, iterations: iter, residual: true_res, converged: true, }
        }
      }
    }
    z = precond.apply(r)
    let rz_new = dot(r, z)
    let beta = rz_new / rz
    if beta.is_nan() || beta.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "cg: numerical breakdown: beta is non-finite at iteration " +
        iter.to_string(),
      )
    }
    // p = z + beta * p
    for i = 0; i < n; i = i + 1 {
      p[i] = z[i] + beta * p[i]
    }
    rz = rz_new
  }
  let final_res = relative_residual(a, x, b)
  if !is_finite_vector(x) {
    raise @err.SparseError::UnsupportedMatrix(
      "cg: numerical breakdown: iterate contains non-finite values",
    )
  }
  {
    x,
    iterations: options.max_iter,
    residual: if final_res.is_nan() || final_res.is_inf() {
      norm2(r) / nb
    } else {
      final_res
    },
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
  initial? : Array[Double]? = None,
) -> SolveResult raise @err.SparseError {
  check_system(a, b)
  check_solver_options(options)
  let mag = system_magnitude(a, b)
  if mag < 1.0e-100 && mag > 0.0 {
    let s = SUBNORMAL_SCALE_FACTOR
    let scaled_a = a.scale(s)
    let scaled_b : Array[Double] = Array::make(b.length(), 0.0)
    for i = 0; i < b.length(); i = i + 1 {
      scaled_b[i] = b[i] * s
    }
    return bicgstab(scaled_a, scaled_b, options~, precond~, initial~)
  }
  let n = a.nrows()
  let x = starting_point(a, initial)
  let nb = norm2(b)
  if nb.is_nan() || nb.is_inf() {
    raise @err.SparseError::UnsupportedMatrix(
      "bicgstab: right-hand side norm is non-finite or overflows floating-point representation",
    )
  }
  if nb == 0.0 {
    return {
      x: Array::make(n, 0.0),
      iterations: 0,
      residual: 0.0,
      converged: true,
    }
  }
  let r = sub_vec(b, a.spmv(x))
  let first_residual = norm2(r) / nb
  if first_residual <= options.tol {
    if is_finite_vector(x) {
      let true_res = relative_residual(a, x, b)
      if is_finite_val(true_res) && true_res <= options.tol {
        return { x, iterations: 0, residual: true_res, converged: true, }
      }
    }
  }
  let r_hat = r.copy()
  let mut p : Array[Double] = Array::make(n, 0.0)
  let mut v : Array[Double] = Array::make(n, 0.0)
  let mut rho_old = 1.0
  let mut alpha = 1.0
  let mut omega = 1.0
  for iter = 1; iter <= options.max_iter; iter = iter + 1 {
    let rho = dot(r_hat, r)
    if rho.is_nan() || rho.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: numerical breakdown: rho is non-finite at iteration " +
        iter.to_string(),
      )
    }
    if rho == 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: numerical breakdown: rho became zero at iteration " +
        iter.to_string() +
        " (underflow)",
      )
    }
    if iter == 1 {
      p = r.copy()
    } else {
      if omega == 0.0 {
        raise @err.SparseError::UnsupportedMatrix(
          "bicgstab: numerical breakdown: omega became zero at iteration " +
          iter.to_string(),
        )
      }
      if !is_finite_val(omega) {
        raise @err.SparseError::UnsupportedMatrix(
          "bicgstab: numerical breakdown: omega is non-finite at iteration " +
          iter.to_string(),
        )
      }
      let beta = rho / rho_old * (alpha / omega)
      if !is_finite_val(beta) {
        raise @err.SparseError::UnsupportedMatrix(
          "bicgstab: numerical breakdown: beta is non-finite at iteration " +
          iter.to_string(),
        )
      }
      // p = r + beta * (p - omega * v)
      for i = 0; i < n; i = i + 1 {
        p[i] = r[i] + beta * (p[i] - omega * v[i])
      }
    }
    let p_hat = precond.apply(p)
    v = a.spmv(p_hat)
    let rhv = dot(r_hat, v)
    if rhv.is_nan() || rhv.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: numerical breakdown: r_hat^T v is non-finite at iteration " +
        iter.to_string(),
      )
    }
    if rhv == 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: numerical breakdown: r_hat^T v became zero at iteration " +
        iter.to_string(),
      )
    }
    alpha = rho / rhv
    if !is_finite_val(alpha) {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: numerical breakdown: alpha is non-finite at iteration " +
        iter.to_string(),
      )
    }
    // s = r - alpha * v
    let s = sub_vec(r, scaled(alpha, v))
    let s_norm = norm2(s) / nb
    if s_norm <= options.tol {
      let x_cand = x.copy()
      axpy(alpha, p_hat, x_cand)
      if is_finite_vector(x_cand) {
        let true_res = relative_residual(a, x_cand, b)
        if is_finite_val(true_res) && true_res <= options.tol {
          return {
            x: x_cand,
            iterations: iter,
            residual: true_res,
            converged: true,
          }
        }
      }
    }
    let s_hat = precond.apply(s)
    let t = a.spmv(s_hat)
    let tt = dot(t, t)
    if tt.is_nan() || tt.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: numerical breakdown: t^T t is non-finite at iteration " +
        iter.to_string(),
      )
    }
    if tt == 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: numerical breakdown: t^T t became zero at iteration " +
        iter.to_string(),
      )
    }
    omega = dot(t, s) / tt
    if !is_finite_val(omega) {
      raise @err.SparseError::UnsupportedMatrix(
        "bicgstab: numerical breakdown: omega is non-finite at iteration " +
        iter.to_string(),
      )
    }
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
      if is_finite_vector(x) {
        let true_res = relative_residual(a, x, b)
        if is_finite_val(true_res) && true_res <= options.tol {
          return { x, iterations: iter, residual: true_res, converged: true, }
        }
      }
    }
    rho_old = rho
  }
  let final_res = relative_residual(a, x, b)
  if !is_finite_vector(x) {
    raise @err.SparseError::UnsupportedMatrix(
      "bicgstab: numerical breakdown: iterate contains non-finite values",
    )
  }
  {
    x,
    iterations: options.max_iter,
    residual: if final_res.is_nan() || final_res.is_inf() {
      norm2(r) / nb
    } else {
      final_res
    },
    converged: false,
  }
}

///|
// Resolve the starting point. A caller who has a previous solution — from the
// last time step, or from a slightly perturbed problem — can hand it in and
// save the iterations it would take to rediscover it.
fn starting_point(
  a : @csr.Csr[Double],
  initial : Array[Double]?,
) -> Array[Double] raise @err.SparseError {
  match initial {
    None => Array::make(a.nrows(), 0.0)
    Some(x0) => {
      if x0.length() != a.nrows() {
        raise @err.SparseError::DimensionMismatch(
          "initial guess has length " +
          x0.length().to_string() +
          ", expected " +
          a.nrows().to_string(),
        )
      }
      for i = 0; i < x0.length(); i = i + 1 {
        let v = x0[i]
        if v.is_nan() || v.is_inf() {
          raise @err.SparseError::UnsupportedMatrix(
            "starting_point: initial guess contains non-finite values",
          )
        }
      }
      x0.copy()
    }
  }
}

///|
/// Restarted GMRES, for general square systems.
///
/// Builds an orthonormal basis of the Krylov subspace and minimises the
/// residual over it, restarting every `restart` iterations to bound the storage
/// at `restart` vectors.
///
/// Preconditioning is applied on the right: the method solves `A M^-1 u = b`
/// and forms `x = x0 + M^-1 u`. Convergence is checked against the estimated
/// residual from Givens rotations and verified against the true relative
/// residual `||b - A x||_2 / ||b||_2` to guarantee that numerical drift or
/// precision limits do not produce false convergence.
pub fn gmres(
  a : @csr.Csr[Double],
  b : Array[Double],
  options? : SolverOptions = SolverOptions::defaults(),
  precond? : Precond = Precond::Identity,
  restart? : Int = 30,
  initial? : Array[Double]? = None,
) -> SolveResult raise @err.SparseError {
  check_system(a, b)
  check_solver_options(options)
  if restart <= 0 {
    raise @err.SparseError::DimensionMismatch(
      "gmres: restart must be positive, got " + restart.to_string(),
    )
  }
  let m64 = restart.to_int64()
  if (m64 + 1L) * m64 > 2147483647L || restart > 46340 {
    raise @err.SparseError::DimensionMismatch(
      "gmres: restart parameter " +
      restart.to_string() +
      " exceeds maximum supported capacity",
    )
  }
  let mag = system_magnitude(a, b)
  if mag < 1.0e-100 && mag > 0.0 {
    let s = SUBNORMAL_SCALE_FACTOR
    let scaled_a = a.scale(s)
    let scaled_b : Array[Double] = Array::make(b.length(), 0.0)
    for i = 0; i < b.length(); i = i + 1 {
      scaled_b[i] = b[i] * s
    }
    return gmres(scaled_a, scaled_b, options~, restart~, precond~, initial~)
  }
  let n = a.nrows()
  let x = starting_point(a, initial)
  let nb = norm2(b)
  if nb.is_nan() || nb.is_inf() {
    raise @err.SparseError::UnsupportedMatrix(
      "gmres: right-hand side norm is non-finite or overflows floating-point representation",
    )
  }
  if nb == 0.0 {
    return {
      x: Array::make(n, 0.0),
      iterations: 0,
      residual: 0.0,
      converged: true,
    }
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
      if is_finite_vector(x) {
        let true_res = relative_residual(a, x, b)
        if is_finite_val(true_res) && true_res <= options.tol {
          return { x, iterations, residual: true_res, converged: true, }
        }
      }
    }
    if beta == 0.0 || !is_finite_val(beta) {
      break
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
      let denom = hypot(h[j + j * (m + 1)], hnext)
      if denom == 0.0 || !is_finite_val(denom) {
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
          // Check candidate update without prematurely modifying the restart base point x.
          let x_cand = x.copy()
          apply_gmres_update(x_cand, basis, h, g, m, j + 1, precond)
          if is_finite_vector(x_cand) {
            let true_res = relative_residual(a, x_cand, b)
            if is_finite_val(true_res) && true_res <= options.tol {
              return {
                x: x_cand,
                iterations,
                residual: true_res,
                converged: true,
              }
            }
          }
        }
        if hnext == 0.0 {
          inner_converged = true
          j = j + 1
          break
        }
      }
      if inner_converged {
        break
      }
      j = j + 1
    }
    // Restart: form the best iterate from the basis built in this cycle.
    let used = j
    if used > 0 {
      let x_cand = x.copy()
      apply_gmres_update(x_cand, basis, h, g, m, used, precond)
      if is_finite_vector(x_cand) {
        for i = 0; i < n; i = i + 1 {
          x[i] = x_cand[i]
        }
      }
    }
    let after = sub_vec(b, a.spmv(x))
    residual = norm2(after) / nb
    if residual <= options.tol && is_finite_vector(x) {
      let true_res = relative_residual(a, x, b)
      if is_finite_val(true_res) && true_res <= options.tol {
        return { x, iterations, residual: true_res, converged: true, }
      }
    }
    if iterations >= options.max_iter {
      break
    }
  }
  let final_res = relative_residual(a, x, b)
  if !is_finite_vector(x) {
    raise @err.SparseError::UnsupportedMatrix(
      "gmres: numerical breakdown: iterate contains non-finite values",
    )
  }
  {
    x,
    iterations,
    residual: if final_res.is_nan() || final_res.is_inf() {
      if residual.is_nan() || residual.is_inf() {
        1.0
      } else {
        residual
      }
    } else {
      final_res
    },
    converged: false,
  }
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
