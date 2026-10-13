///|
fn rejected(f : () -> Unit raise @err.SparseError) -> Bool {
  try {
    f()
    false
  } catch {
    _ => true
  }
}

///|
fn no_false_success(
  f : () -> @solve.SolveResult raise @err.SparseError,
  a : @csr.Csr[Double],
  b : Array[Double],
) -> Unit raise {
  let result = Some(f()) catch { _ => None }
  if result is Some(r) && r.converged {
    let actual = @solve.relative_residual(a, r.x, b)
    assert_true(actual >= 0.0 && actual <= 1.0e-10)
    for v in r.x {
      assert_true(!v.is_nan() && !v.is_inf())
    }
  }
}

///|
test "r2 norm2 NaN must not be zero" {
  assert_true(@solve.norm2([0.0 / 0.0]).is_nan())
}

///|
test "r2 norm2 infinity should not become NaN" {
  assert_eq(@solve.norm2([@double.infinity]), @double.infinity)
}

///|
test "r2 finite RHS norm overflow CG must not false converge" {
  let a = @ops.identity(2, 1.0)
  let b = [1.3e308, 1.3e308]
  let result = Some(@solve.cg(a, b, initial=Some([1.0e308, 1.0e308]))) catch {
    _ => None
  }
  if result is Some(r) {
    assert_true(!r.converged || r.x == b)
  }
}

///|
test "r2 finite RHS norm overflow BiCGSTAB must not false converge" {
  let a = @ops.identity(2, 1.0)
  let b = [1.3e308, 1.3e308]
  let result = Some(@solve.bicgstab(a, b, initial=Some([1.0e308, 1.0e308]))) catch {
    _ => None
  }
  if result is Some(r) {
    assert_true(!r.converged || r.x == b)
  }
}

///|
test "r2 finite RHS norm overflow GMRES must not false converge" {
  let a = @ops.identity(2, 1.0)
  let b = [1.3e308, 1.3e308]
  let result = Some(@solve.gmres(a, b, initial=Some([1.0e308, 1.0e308]))) catch {
    _ => None
  }
  if result is Some(r) {
    assert_true(!r.converged || r.x == b)
  }
}

///|
test "r2 GMRES Givens overflow must not false converge" {
  let a = @ops.identity(1, 1.0e200)
  no_false_success(() => @solve.gmres(a, [1.0]), a, [1.0])
}

///|
test "r2 NaN matrix CG must not false converge" {
  let a = @ops.identity(1, 0.0 / 0.0)
  let result = Some(@solve.cg(a, [1.0])) catch { _ => None }
  if result is Some(r) {
    assert_true(!r.converged)
  }
}

///|
test "r2 NaN initial GMRES must not false converge" {
  let a = @ops.identity(1, 1.0)
  let result = Some(@solve.gmres(a, [1.0], initial=Some([0.0 / 0.0]))) catch {
    _ => None
  }
  if result is Some(r) {
    assert_true(!r.converged)
  }
}

///|
test "r2 integer exponent fraction must be rejected" {
  assert_true(
    rejected(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix array integer general\n1 1\n1e-1\n",
      )
    }),
  )
}

///|
test "r2 integer infinity must be rejected" {
  assert_true(
    rejected(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate integer general\n1 1 1\n1 1 inf\n",
      )
    }),
  )
}

///|
test "r2 coordinate extra fields must be rejected" {
  assert_true(
    rejected(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 2 99\n",
      )
    }),
  )
}

///|
test "r2 MinPlus saturation associativity" {
  let a = @semiring.MinPlus::new(6.0e307)
  let b = @semiring.MinPlus::new(6.0e307)
  let c = @semiring.MinPlus::new(-6.0e307)
  assert_eq((a * b * c).value(), (a * (b * c)).value())
}

///|
test "r2 MinPlus multiple zero representations additive commutativity" {
  let a = @semiring.MinPlus::infinity()
  let b = @semiring.MinPlus::new(1.5e308)
  assert_eq(a + b, b + a)
}

///|
test "r2 MinPlus negative overflow" {
  let a = @semiring.MinPlus::new(-1.0e308)
  let b = @semiring.MinPlus::new(-1.0e308)
  let c = @semiring.MinPlus::new(9.0e307)
  assert_eq((a * b * c).value(), (a * (b * c)).value())
}

///|
test "r2 zero edge invalidates documented component spectral correspondence" {
  let a = @graph.weighted_adjacency_from_edges(2, [(0, 1, 0.0)])
  let n = @graph.normalized_laplacian(a)
  assert_eq(n.to_dense(), [0.0, 0.0, 0.0, 0.0])
  assert_eq(@graph.component_count(a), 2)
}

///|
test "r2 normalized subnormal single edge" {
  let a = @graph.weighted_adjacency_from_edges(2, [(0, 1, 5.0e-324)])
  let n = @graph.normalized_laplacian(a)
  assert_true(Double::abs(n.get_or(0, 1, 0.0) + 1.0) < 1.0e-14)
}

///|
test "r2 normalized isolated one option preserved" {
  let a = @graph.adjacency_from_edges(1, [])
  assert_eq(@graph.normalized_laplacian(a, isolated_zero=false).diagonal(), [
    1.0,
  ])
}

///|
test "r2 all small CG BiCGSTAB must not claim false success" {
  let a = @ops.identity(1, 1.0)
  no_false_success(() => @solve.cg(a, [1.0e-200]), a, [1.0e-200])
  no_false_success(() => @solve.bicgstab(a, [1.0e-200]), a, [1.0e-200])
}

///|
test "r2 finite large CG RHS must not return NaN successful solution" {
  let a = @ops.identity(1, 1.0)
  no_false_success(() => @solve.cg(a, [1.0e200]), a, [1.0e200])
}

///|
test "r2 finite large BiCGSTAB RHS must not return NaN successful solution" {
  let a = @ops.identity(1, 1.0)
  no_false_success(() => @solve.bicgstab(a, [1.0e200]), a, [1.0e200])
}

///|
test "r2 normalized wide finite weights preserves symmetric edge" {
  let a = @graph.weighted_adjacency_from_edges(3, [
    (0, 1, 1.0e-200),
    (0, 2, 1.0e300),
  ])
  let n = @graph.normalized_laplacian(a)
  let u = n.get_or(0, 1, 0.0)
  let v = n.get_or(1, 0, 0.0)
  assert_true(u < 0.0 && v < 0.0)
  assert_true(Double::abs(u / v - 1.0) < 1.0e-14)
}

///|
test "r2 all NaN residual helper must not certify zero" {
  let a = @ops.identity(1, 1.0)
  let r = @solve.relative_residual(a, [0.0 / 0.0], [1.0])
  assert_true(r.is_nan())
}
