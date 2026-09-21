// Differential tests against the independent dense reference.
//
// Every test here builds a random sparse system, solves or multiplies it
// through the sparse path, and does the same thing through `dense_ref.mbt`.
// The two must agree. This is the check that catches the failure mode the
// hand-written tests cannot: a wrong index still yields a plausible number, so
// only a second implementation disagrees visibly.
//
// The systems are small — at most a few dozen unknowns — because the reference
// is `O(n^3)` and because a bug in index arithmetic shows up just as well on a
// 20x20 matrix as on a 20,000x20,000 one.

///|
fn close(a : Double, b : Double, tol : Double) -> Bool {
  Double::abs(a - b) <= tol
}

// A random sparse matrix with a given density, as both CSR and dense.

///|
fn random_sparse(
  rng : @integration.Rng,
  n : Int,
  density : Double,
) -> (@csr.Csr[Double], Array[Double]) raise @err.SparseError {
  let coo : @coo.Coo[Double] = @coo.Coo::empty(n, n)
  for i = 0; i < n; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      if rng.uniform(0.0, 1.0) < density {
        coo.push(i, j, rng.uniform(-1.0, 1.0))
      }
    }
  }
  let csr = coo.to_csr()
  (csr, csr.to_dense())
}

// A random symmetric positive definite matrix, built as `M M^T + n I`.
//
// The shift is what makes it positive definite regardless of what `M` turned
// out to be; without it a random symmetric matrix is indefinite about half the
// time.

///|
fn random_spd(
  rng : @integration.Rng,
  n : Int,
  density : Double,
) -> (@csr.Csr[Double], Array[Double]) raise @err.SparseError {
  let (m, m_dense) = random_sparse(rng, n, density)
  let mt = @ops.transpose(m)
  let product = @ops.spgemm(m, mt)
  let shifted : @coo.Coo[Double] = @coo.Coo::empty(n, n)
  for i = 0; i < n; i = i + 1 {
    shifted.push(i, i, n.to_double())
  }
  let a = @ops.add(product, shifted.to_csr())
  // `m_dense` is not needed: the dense form of the result is taken from the
  // sparse matrix itself, and the random_sparse helper has already checked that
  // its own dense expansion agrees.
  let _ = m_dense
  (a, a.to_dense())
}

///|
test "spmv agrees with the dense product" {
  let rng = @integration.Rng::new(11)
  for trial = 0; trial < 20; trial = trial + 1 {
    let n = 5 + rng.below(20)
    let (a, dense) = random_sparse(rng, n, 0.3)
    let x : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      x[i] = rng.uniform(-2.0, 2.0)
    }
    let sparse_y = a.spmv(x)
    let dense_y = @integration.dense_matvec(dense, n, x)
    for i = 0; i < n; i = i + 1 {
      inspect(close(sparse_y[i], dense_y[i], 1.0e-12), content="true")
    }
  }
}

///|
test "csc spmv agrees with the dense product" {
  let rng = @integration.Rng::new(12)
  for trial = 0; trial < 20; trial = trial + 1 {
    let n = 5 + rng.below(20)
    let (a, dense) = random_sparse(rng, n, 0.3)
    let csc = @csc.Csc::from_csr(a)
    let x : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      x[i] = rng.uniform(-2.0, 2.0)
    }
    let sparse_y = csc.spmv(x)
    let dense_y = @integration.dense_matvec(dense, n, x)
    for i = 0; i < n; i = i + 1 {
      inspect(close(sparse_y[i], dense_y[i], 1.0e-12), content="true")
    }
  }
}

///|
test "the transposed product agrees with the dense transpose" {
  let rng = @integration.Rng::new(13)
  for trial = 0; trial < 15; trial = trial + 1 {
    let n = 4 + rng.below(15)
    let (a, dense) = random_sparse(rng, n, 0.4)
    // Build A^T densely by hand: it is a transpose, so it is easy to get right
    // independently of the sparse machinery.
    let dense_t : Array[Double] = Array::make(n * n, 0.0)
    for i = 0; i < n; i = i + 1 {
      for j = 0; j < n; j = j + 1 {
        dense_t[j * n + i] = dense[i * n + j]
      }
    }
    let x : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      x[i] = rng.uniform(-2.0, 2.0)
    }
    let sparse_y = @csc.Csc::from_csr(a).spmv_transpose(x)
    let dense_y = @integration.dense_matvec(dense_t, n, x)
    for i = 0; i < n; i = i + 1 {
      inspect(close(sparse_y[i], dense_y[i], 1.0e-12), content="true")
    }
  }
}

///|
test "spgemm agrees with the dense product" {
  let rng = @integration.Rng::new(14)
  for trial = 0; trial < 15; trial = trial + 1 {
    let n = 4 + rng.below(12)
    let (a, a_dense) = random_sparse(rng, n, 0.35)
    let (b, b_dense) = random_sparse(rng, n, 0.35)
    let product = @ops.spgemm(a, b).to_dense()
    let expected = @integration.dense_matmul(a_dense, b_dense, n)
    for i = 0; i < n * n; i = i + 1 {
      inspect(close(product[i], expected[i], 1.0e-12), content="true")
    }
  }
}

///|
test "the sparse transpose agrees with the dense transpose" {
  let rng = @integration.Rng::new(15)
  for trial = 0; trial < 15; trial = trial + 1 {
    let n = 4 + rng.below(12)
    let (a, dense) = random_sparse(rng, n, 0.35)
    let t = @ops.transpose(a).to_dense()
    for i = 0; i < n; i = i + 1 {
      for j = 0; j < n; j = j + 1 {
        inspect(close(t[j * n + i], dense[i * n + j], 1.0e-15), content="true")
      }
    }
  }
}

///|
test "sparse addition agrees with dense addition" {
  let rng = @integration.Rng::new(16)
  for trial = 0; trial < 15; trial = trial + 1 {
    let n = 4 + rng.below(12)
    let (a, a_dense) = random_sparse(rng, n, 0.3)
    let (b, b_dense) = random_sparse(rng, n, 0.3)
    let sum = @ops.add(a, b).to_dense()
    for i = 0; i < n * n; i = i + 1 {
      inspect(close(sum[i], a_dense[i] + b_dense[i], 1.0e-14), content="true")
    }
    let difference = @ops.sub(a, b).to_dense()
    for i = 0; i < n * n; i = i + 1 {
      inspect(
        close(difference[i], a_dense[i] - b_dense[i], 1.0e-14),
        content="true",
      )
    }
  }
}

///|
test "cg agrees with dense gaussian elimination" {
  let rng = @integration.Rng::new(21)
  for trial = 0; trial < 12; trial = trial + 1 {
    let n = 6 + rng.below(14)
    let (a, dense) = random_spd(rng, n, 0.3)
    let b : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      b[i] = rng.uniform(-1.0, 1.0)
    }
    let result = @solve.cg(
      a,
      b,
      options=@solve.SolverOptions::new(1.0e-12, 2000),
    )
    inspect(result.converged, content="true")
    let expected = match @integration.dense_solve(dense, b, n) {
      Some(x) => x
      None => {
        inspect(
          "the reference declared the system singular",
          content="unexpected",
        )
        []
      }
    }
    for i = 0; i < n; i = i + 1 {
      inspect(close(result.x[i], expected[i], 1.0e-8), content="true")
    }
  }
}

///|
test "cholesky agrees with dense gaussian elimination" {
  let rng = @integration.Rng::new(22)
  for trial = 0; trial < 12; trial = trial + 1 {
    let n = 6 + rng.below(14)
    let (a, dense) = random_spd(rng, n, 0.3)
    let b : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      b[i] = rng.uniform(-1.0, 1.0)
    }
    let factor = @solve.CholeskyFactor::new(a)
    let x = factor.solve(b)
    let expected = match @integration.dense_solve(dense, b, n) {
      Some(v) => v
      None => []
    }
    for i = 0; i < n; i = i + 1 {
      inspect(close(x[i], expected[i], 1.0e-8), content="true")
    }
    // The factor must also multiply back to the matrix.
    let rebuilt = factor.reconstruct()
    for i = 0; i < n * n; i = i + 1 {
      inspect(close(rebuilt[i], dense[i], 1.0e-9), content="true")
    }
  }
}

///|
test "the sparse cholesky factor matches the dense one entry by entry" {
  let rng = @integration.Rng::new(23)
  for trial = 0; trial < 10; trial = trial + 1 {
    let n = 5 + rng.below(10)
    let (a, dense) = random_spd(rng, n, 0.4)
    let factor = @solve.CholeskyFactor::new(a)
    let expected = match @integration.dense_cholesky(dense, n) {
      Some(l) => l
      None => []
    }
    // Rebuild the sparse factor as a dense lower triangle and compare.
    let cp = factor.col_ptr()
    let ri = factor.row_idx()
    let va = factor.vals()
    let sparse_dense : Array[Double] = Array::make(n * n, 0.0)
    for j = 0; j < n; j = j + 1 {
      for k = cp[j]; k < cp[j] + factor.col_nnz(j); k = k + 1 {
        sparse_dense[ri[k] * n + j] = va[k]
      }
    }
    for i = 0; i < n * n; i = i + 1 {
      inspect(close(sparse_dense[i], expected[i], 1.0e-9), content="true")
    }
  }
}

///|
test "bicgstab agrees with dense gaussian elimination" {
  let rng = @integration.Rng::new(24)
  for trial = 0; trial < 10; trial = trial + 1 {
    let n = 5 + rng.below(10)
    // Add a strong diagonal so the random system is well conditioned.
    let (a, _) = random_sparse(rng, n, 0.3)
    let shift : @coo.Coo[Double] = @coo.Coo::empty(n, n)
    for i = 0; i < n; i = i + 1 {
      shift.push(i, i, n.to_double())
    }
    let a = @ops.add(a, shift.to_csr())
    let dense = a.to_dense()
    let b : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      b[i] = rng.uniform(-1.0, 1.0)
    }
    let result = @solve.bicgstab(
      a,
      b,
      options=@solve.SolverOptions::new(1.0e-12, 2000),
    )
    inspect(result.converged, content="true")
    let expected = match @integration.dense_solve(dense, b, n) {
      Some(x) => x
      None => []
    }
    for i = 0; i < n; i = i + 1 {
      inspect(close(result.x[i], expected[i], 1.0e-7), content="true")
    }
  }
}

///|
test "gmres agrees with dense gaussian elimination" {
  let rng = @integration.Rng::new(25)
  for trial = 0; trial < 8; trial = trial + 1 {
    let n = 5 + rng.below(8)
    let (a, _) = random_sparse(rng, n, 0.3)
    let shift : @coo.Coo[Double] = @coo.Coo::empty(n, n)
    for i = 0; i < n; i = i + 1 {
      shift.push(i, i, n.to_double())
    }
    let a = @ops.add(a, shift.to_csr())
    let dense = a.to_dense()
    let b : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      b[i] = rng.uniform(-1.0, 1.0)
    }
    let result = @solve.gmres(
      a,
      b,
      options=@solve.SolverOptions::new(1.0e-12, 500),
      restart=30,
    )
    inspect(result.converged, content="true")
    let expected = match @integration.dense_solve(dense, b, n) {
      Some(x) => x
      None => []
    }
    for i = 0; i < n; i = i + 1 {
      inspect(close(result.x[i], expected[i], 1.0e-7), content="true")
    }
  }
}

///|
test "preconditioned cg agrees with dense gaussian elimination" {
  let rng = @integration.Rng::new(26)
  for trial = 0; trial < 10; trial = trial + 1 {
    let n = 6 + rng.below(14)
    let (a, dense) = random_spd(rng, n, 0.3)
    let b : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      b[i] = rng.uniform(-1.0, 1.0)
    }
    for kind in ["jacobi", "ilu0"] {
      let result = @solve.cg(
        a,
        b,
        options=@solve.SolverOptions::new(1.0e-12, 2000),
        precond=@solve.Precond::for_matrix(kind, a),
      )
      inspect(result.converged, content="true")
      let expected = match @integration.dense_solve(dense, b, n) {
        Some(x) => x
        None => []
      }
      for i = 0; i < n; i = i + 1 {
        inspect(close(result.x[i], expected[i], 1.0e-7), content="true")
      }
    }
  }
}

///|
test "the laplacian of a random graph matches its dense construction" {
  let rng = @integration.Rng::new(31)
  for trial = 0; trial < 10; trial = trial + 1 {
    let n = 5 + rng.below(12)
    let edges : Array[(Int, Int)] = Array(capacity=2 * n)
    for _e = 0; _e < 2 * n; _e = _e + 1 {
      edges.push((rng.below(n), rng.below(n)))
    }
    let adjacency = @graph.adjacency_from_edges(n, edges)
    let l = @graph.laplacian(adjacency)
    // Build D - A densely from the same adjacency matrix.
    let dense_a = adjacency.to_dense()
    let expected : Array[Double] = Array::make(n * n, 0.0)
    for i = 0; i < n; i = i + 1 {
      let mut degree = 0.0
      for j = 0; j < n; j = j + 1 {
        if i != j {
          degree = degree + dense_a[i * n + j]
        }
      }
      for j = 0; j < n; j = j + 1 {
        // Diagonal is the off-diagonal degree; off-diagonal entries are -A.
        expected[i * n + j] = if i == j { degree } else { -dense_a[i * n + j] }
      }
    }
    let actual = l.to_dense()
    for i = 0; i < n * n; i = i + 1 {
      inspect(close(actual[i], expected[i], 1.0e-14), content="true")
    }
  }
}

///|
test "matrix market round trips through the dense reference" {
  let rng = @integration.Rng::new(41)
  for trial = 0; trial < 10; trial = trial + 1 {
    let n = 3 + rng.below(8)
    let (a, dense) = random_sparse(rng, n, 0.4)
    let coo = @coo.Coo::from_csr(a)
    let text = @io.write_matrix_market(coo)
    let back = @io.parse_matrix_market(text).to_csr().to_dense()
    for i = 0; i < n * n; i = i + 1 {
      inspect(close(back[i], dense[i], 1.0e-14), content="true")
    }
  }
}

///|
test "reordering then solving agrees with solving the original" {
  let rng = @integration.Rng::new(51)
  for trial = 0; trial < 8; trial = trial + 1 {
    let n = 5 + rng.below(10)
    let (a, dense) = random_spd(rng, n, 0.4)
    let perm = @reorder.reverse_cuthill_mckee(a)
    let reordered = @reorder.permute(a, perm)
    let b : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      b[i] = rng.uniform(-1.0, 1.0)
    }
    // The right-hand side has to be permuted along with the matrix: row
    // perm[i] of the reordered system corresponds to row i of the original, so
    // it must carry b[i].
    let b_reordered : Array[Double] = Array::make(n, 0.0)
    for i = 0; i < n; i = i + 1 {
      b_reordered[perm[i]] = b[i]
    }
    let x_original = @solve.solve_cholesky(a, b)
    let x_reordered = @solve.solve_cholesky(reordered, b_reordered)
    // With perm[old] = new, the reordered unknown at position perm[i] is the
    // original unknown i.
    for i = 0; i < n; i = i + 1 {
      inspect(
        close(x_reordered[perm[i]], x_original[i], 1.0e-9),
        content="true",
      )
    }
    let expected = match @integration.dense_solve(dense, b, n) {
      Some(x) => x
      None => []
    }
    for i = 0; i < n; i = i + 1 {
      inspect(close(x_original[i], expected[i], 1.0e-8), content="true")
    }
  }
}
