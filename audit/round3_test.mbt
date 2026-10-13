///|
test "r3 MinPlus ordinary negative constructor preserves value" {
  assert_eq(@semiring.MinPlus::new(-1.0).value(), -1.0)
}

///|
test "r3 MinPlus ordinary negative path sum" {
  let a = @semiring.MinPlus::new(-1.0)
  let b = @semiring.MinPlus::new(2.0)
  assert_eq((a * b).value(), 1.0)
}

///|
test "r3 MinPlus distinct ordinary negative values remain distinct" {
  assert_true(@semiring.MinPlus::new(-1.0) != @semiring.MinPlus::new(-2.0))
}

///|
test "r3 MinPlus negative infinity multiplicative identity" {
  let a = @semiring.MinPlus::new(-@double.infinity)
  assert_eq((a * @semiring.MinPlus::one()).value(), -@double.infinity)
}

///|
test "r3 zero count skew array rejects extra numerical value" {
  assert_true(
    rejected(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix array real skew-symmetric\n1 1\n7\n",
      )
    }),
  )
}

///|
test "r3 zero count array rejects extra junk" {
  assert_true(
    rejected(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix array real general\n0 2\njunk\n",
      )
    }),
  )
}

///|
test "r3 valid empty skew array comments still accepted" {
  let a = @io.parse_matrix_market(
    "%%MatrixMarket matrix array real skew-symmetric\n1 1\n% trailing comment\n\n",
  )
  assert_eq(a.nnz(), 0)
}

///|
test "r3 observations" {
  let a = @semiring.MinPlus::new(-1.0)
  let b = @semiring.MinPlus::new(2.0)
  println("new(-1) value=\{a.value()}, (-1)*2 value=\{(a * b).value()}")
  let z = @semiring.MinPlus::new(-@double.infinity) * @semiring.MinPlus::one()
  println("negative infinity times one value=\{z.value()}")
}

///|
test "r3 GMRES rejected candidate must not be applied twice" {
  let a = @ops.identity(1, 49.0)
  let opts = @solve.SolverOptions::new(1.0e-17, 1)
  let r = @solve.gmres(a, [1.0], options=opts, restart=1)
  println(
    "GMRES A=49 b=1 tol=1e-17 max_iter=1: x=\{r.x[0]}, residual=\{r.residual}, converged=\{r.converged}",
  )
  assert_true(r.residual <= 1.0e-14)
}

///|
test "r3 infinite tolerance must not certify infinite residual" {
  let a = @ops.identity(1, 2.0)
  assert_true(
    rejected(() => {
      let _ = @solve.SolverOptions::new(@double.infinity, 1)
    }),
  )
  assert_true(
    rejected(() => {
      let _ = @solve.gmres(
        a,
        [1.0],
        options={ tol: @double.infinity, max_iter: 1, },
        restart=1,
        initial=Some([-1.0e308]),
      )
    }),
  )
}

///|
test "r3 COO and CSC dimension bounds enforce contract without abort" {
  assert_true(
    rejected(() => {
      let _ : @coo.Coo[Double] = @coo.Coo::empty(2147483647, 0)
    }),
  )
  assert_true(
    rejected(() => {
      let _ : @coo.Coo[Double] = @coo.Coo::empty(100_000_001, 1)
    }),
  )
  assert_true(
    rejected(() => {
      let _ : @csc.Csc[Double] = @csc.Csc::new(0, 2147483647, [0], [], [])
    }),
  )
  assert_true(
    rejected(() => {
      let _ : @csr.Csr[Double] = @csr.Csr::new(2147483647, 0, [0], [], [])
    }),
  )
}

///|
test "r3 MinPlus negative weight path SpGEMM" {
  // 3-node graph with negative edge weights:
  // 0 -> 1 (-2.0), 1 -> 2 (-3.0), 0 -> 2 (0.0)
  let coo : @coo.Coo[@semiring.MinPlus] = @coo.Coo::empty(3, 3)
  coo.push(0, 1, @semiring.MinPlus::new(-2.0))
  coo.push(1, 2, @semiring.MinPlus::new(-3.0))
  coo.push(0, 2, @semiring.MinPlus::new(0.0))
  let a = coo.to_csr()
  let a2 = @ops.spgemm(a, a)
  // Two-hop cost 0 -> 1 -> 2 should be -2.0 + (-3.0) = -5.0
  let two_hop = a2.get_or(0, 2, @semiring.MinPlus::infinity()).value()
  assert_eq(two_hop, -5.0)
}

///|
test "r3 SolverOptions input validation" {
  assert_true(
    rejected(() => {
      let _ = @solve.SolverOptions::new(-1.0, 100)
    }),
  )
  assert_true(
    rejected(() => {
      let _ = @solve.SolverOptions::new(0.0, 100)
    }),
  )
  assert_true(
    rejected(() => {
      let _ = @solve.SolverOptions::new(@double.infinity, 100)
    }),
  )
  assert_true(
    rejected(() => {
      let _ = @solve.SolverOptions::new(0.0 / 0.0, 100)
    }),
  )
  assert_true(
    rejected(() => {
      let _ = @solve.SolverOptions::new(1.0e-5, 0)
    }),
  )
  assert_true(
    rejected(() => {
      let _ = @solve.SolverOptions::new(1.0e-5, -1)
    }),
  )
}

///|
test "r3 GMRES restart parameter limits" {
  let a = @ops.identity(2, 1.0)
  assert_true(
    rejected(() => {
      let _ = @solve.gmres(a, [1.0, 1.0], restart=0)
    }),
  )
  assert_true(
    rejected(() => {
      let _ = @solve.gmres(a, [1.0, 1.0], restart=-5)
    }),
  )
  assert_true(
    rejected(() => {
      let _ = @solve.gmres(a, [1.0, 1.0], restart=50000)
    }),
  )
}

///|
test "r3 Solvers positive convergence on well-conditioned systems" {
  // 1. CG on 2x2 SPD matrix: [4 1; 1 3] x = [1; 2] => x = [1/11, 7/11]
  let coo_spd : @coo.Coo[Double] = @coo.Coo::empty(2, 2)
  coo_spd.push(0, 0, 4.0)
  coo_spd.push(0, 1, 1.0)
  coo_spd.push(1, 0, 1.0)
  coo_spd.push(1, 1, 3.0)
  let a_spd = coo_spd.to_csr()
  let b_spd = [1.0, 2.0]
  let opts = @solve.SolverOptions::new(1.0e-12, 100)
  let r_cg = @solve.cg(a_spd, b_spd, options=opts)
  assert_true(r_cg.converged)
  assert_true(r_cg.residual <= 1.0e-12)
  assert_true(Double::abs(r_cg.x[0] - 1.0 / 11.0) < 1.0e-10)
  assert_true(Double::abs(r_cg.x[1] - 7.0 / 11.0) < 1.0e-10)

  // 2. BiCGSTAB on 2x2 non-symmetric matrix: [2 1; -1 3] x = [1; 2] => x = [1/7, 5/7]
  let coo_ns : @coo.Coo[Double] = @coo.Coo::empty(2, 2)
  coo_ns.push(0, 0, 2.0)
  coo_ns.push(0, 1, 1.0)
  coo_ns.push(1, 0, -1.0)
  coo_ns.push(1, 1, 3.0)
  let a_ns = coo_ns.to_csr()
  let b_ns = [1.0, 2.0]
  let r_bicg = @solve.bicgstab(a_ns, b_ns, options=opts)
  assert_true(r_bicg.converged)
  assert_true(r_bicg.residual <= 1.0e-12)
  assert_true(Double::abs(r_bicg.x[0] - 1.0 / 7.0) < 1.0e-10)
  assert_true(Double::abs(r_bicg.x[1] - 5.0 / 7.0) < 1.0e-10)

  // 3. GMRES on 2x2 matrix with restart and initial guess: [10 1; 2 8] x = [11; 10] => x = [1, 1]
  let coo_gm : @coo.Coo[Double] = @coo.Coo::empty(2, 2)
  coo_gm.push(0, 0, 10.0)
  coo_gm.push(0, 1, 1.0)
  coo_gm.push(1, 0, 2.0)
  coo_gm.push(1, 1, 8.0)
  let a_gm = coo_gm.to_csr()
  let b_gm = [11.0, 10.0]
  let r_gmres = @solve.gmres(
    a_gm,
    b_gm,
    options=opts,
    restart=2,
    initial=Some([0.5, 0.5]),
  )
  assert_true(r_gmres.converged)
  assert_true(r_gmres.residual <= 1.0e-12)
  assert_true(Double::abs(r_gmres.x[0] - 1.0) < 1.0e-10)
  assert_true(Double::abs(r_gmres.x[1] - 1.0) < 1.0e-10)
}
