///|
fn r4_rejected(f : () -> Unit raise @err.SparseError) -> Bool {
  try {
    f()
    false
  } catch {
    _ => true
  }
}

///|
test "r4 MinPlus min orders generated mixed sign components numerically" {
  let a = @semiring.MinPlus::new(1.0e300) * @semiring.MinPlus::new(-7.5e299)
  let b = @semiring.MinPlus::new(5.0e299)
  println(
    "min mixed representation: a=\{a.value()}, b=\{b.value()}, min=\{(a+b).value()}",
  )
  assert_eq((a + b).value(), a.value())
}

///|
test "r4 MinPlus distributivity with generated mixed components" {
  let a = @semiring.MinPlus::new(1.0e300) * @semiring.MinPlus::new(-7.5e299)
  let b = @semiring.MinPlus::new(5.0e299)
  let c = @semiring.MinPlus::new(7.5e299)
  let lhs = ((a + b) * c).value()
  let rhs = (a * c + b * c).value()
  println("distributivity lhs=\{lhs}, rhs=\{rhs}")
  assert_eq(lhs, rhs)
}

///|
test "r4 documented floating associativity observation" {
  let a = @semiring.MinPlus::new(1.0e16)
  let b = @semiring.MinPlus::new(-1.0e16)
  let c = @semiring.MinPlus::new(1.0)
  println(
    "ordinary associative regrouping: lhs=\{((a*b)*c).value()}, rhs=\{(a*(b*c)).value()}",
  )
}

///|
test "r4 subnormal GMRES warm start must not falsely certify 25 percent error" {
  let a = @ops.identity(1, 5.0e-324)
  let r = @solve.gmres(a, [5.0e-324], initial=Some([0.75]))
  println(
    "subnormal gmres x=\{r.x[0]}, residual=\{r.residual}, success=\{r.converged}",
  )
  assert_true(!r.converged || Double::abs(r.x[0] - 1.0) <= 1.0e-10)
}

///|
test "r4 subnormal CG warm start must not falsely certify 25 percent error" {
  let a = @ops.identity(1, 5.0e-324)
  let r = @solve.cg(a, [5.0e-324], initial=Some([0.75]))
  println(
    "subnormal cg x=\{r.x[0]}, residual=\{r.residual}, success=\{r.converged}",
  )
  assert_true(!r.converged || Double::abs(r.x[0] - 1.0) <= 1.0e-10)
}

///|
test "r4 subnormal BiCGSTAB warm start must not falsely certify 25 percent error" {
  let a = @ops.identity(1, 5.0e-324)
  let r = @solve.bicgstab(a, [5.0e-324], initial=Some([0.75]))
  println(
    "subnormal bicgstab x=\{r.x[0]}, residual=\{r.residual}, success=\{r.converged}",
  )
  assert_true(!r.converged || Double::abs(r.x[0] - 1.0) <= 1.0e-10)
}

///|
test "r4 zero array CRLF whitespace comment combinations accepted" {
  let a = @io.parse_matrix_market(
    "%%MatrixMarket matrix array real skew-symmetric\r\n\t %before\r\n1\t1\r\n \t \r\n%after\r\n  %after2\r\n\r\n",
  )
  assert_eq(a.nnz(), 0)
}

///|
test "r4 zero coordinate cases and extra fields" {
  let a = @io.parse_matrix_market(
    "%%MatrixMarket matrix coordinate real general\r\n0 2 0\r\n % tail\r\n\r\n",
  )
  assert_eq(a.nnz(), 0)
  assert_true(
    r4_rejected(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n0 2 1\n1 1 3\n",
      )
    }),
  )
  assert_true(
    r4_rejected(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 3 4\n",
      )
    }),
  )
}

///|
test "r4 singular inconsistent Krylov must not return successful result" {
  let a = @ops.identity(1, 0.0)
  let funcs : Array[() -> @solve.SolveResult raise @err.SparseError] = [
    () => @solve.cg(a, [1.0]),
    () => @solve.bicgstab(a, [1.0]),
    () => @solve.gmres(a, [1.0], options=@solve.SolverOptions::new(1.0e-10, 2)),
  ]
  for f in funcs {
    let r = Some(f()) catch { _ => None }
    if r is Some(r) {
      assert_true(!r.converged)
    }
  }
}

///|
test "r4 GMRES tiny finite nonsingular system must not return invalid failed iterate" {
  let a = @ops.identity(1, 1.0e-320)
  let r = Some(
    @solve.gmres(a, [1.0], options=@solve.SolverOptions::new(1.0e-10, 1)),
  ) catch {
    _ => None
  }
  if r is Some(r) {
    println(
      "GMRES tiny coefficient x=\{r.x[0]}, residual=\{r.residual}, success=\{r.converged}",
    )
    assert_true(@solve.is_finite_vector(r.x))
  }
}

///|
test "r4 PageRank tiny self loop scale invariance" {
  let a = @ops.identity(1, 5.0e-324)
  let r = @graph.page_rank(a)
  println("PageRank tiny self-loop rank=\{r[0]}")
  assert_eq(r, [1.0])
}

///|
test "r4 PageRank large row sum scale invariance" {
  let a = @ops.from_dense([1.0e308, 1.0e308, 0.0, 1.0e308], 2, 2, x => x != 0.0)
  let b = @ops.from_dense([1.0, 1.0, 0.0, 1.0], 2, 2, x => x != 0.0)
  let ra = @graph.page_rank(a)
  let rb = @graph.page_rank(b)
  println("PageRank overflow row scaled: a0=\{ra[0]}, b0=\{rb[0]}")
  assert_true(Double::abs(ra[0] - rb[0]) < 1.0e-10)
}

///|
test "r4 SpMM output size overflow must be checked" {
  let a : @csr.Csr[Double] = @csr.Csr::new(
    65536,
    0,
    Array::make(65537, 0),
    [],
    [],
  )
  let r = Some(@ops.spmm(a, [], 65536)) catch { _ => None }
  if r is Some(r) {
    println("SpMM 65536x65536 output length=\{r.length()}")
    assert_true(false)
  }
}

///|
test "r4 MinPlus actual SpGEMM competing paths choose shorter" {
  let v = @semiring.MinPlus::new(1.0e300) * @semiring.MinPlus::new(-7.5e299)
  let a : @csr.Csr[@semiring.MinPlus] = @csr.Csr::new(1, 2, [0, 2], [0, 1], [
    v,
    @semiring.MinPlus::new(5.0e299),
  ])
  let b : @csr.Csr[@semiring.MinPlus] = @csr.Csr::new(2, 1, [0, 1, 2], [0, 0], [
    @semiring.MinPlus::one(),
    @semiring.MinPlus::one(),
  ])
  let r = @ops.spgemm(a, b).get_or(0, 0, @semiring.MinPlus::infinity()).value()
  println("SpGEMM competing tropical paths result=\{r}")
  assert_eq(r, v.value())
}

///|
test "r4 CG indefinite and singular observations" {
  let ind = @ops.from_dense([1.0, 0.0, 0.0, -1.0], 2, 2, x => x != 0.0)
  let sing = @ops.from_dense([1.0, 0.0, 0.0, 0.0], 2, 2, x => x != 0.0)
  let r1 = @solve.cg(ind, [1.0, 0.0])
  let r2 = @solve.cg(sing, [1.0, 0.0])
  assert_true(r1.converged && r2.converged)
  assert_eq(r1.x, [1.0, 0.0])
  assert_eq(r2.x, [1.0, 0.0])
}

///|
test "r4 subnormal and signed zero constructor control" {
  assert_eq(@semiring.MinPlus::new(5.0e-324).value(), 5.0e-324)
  assert_eq(@semiring.MinPlus::new(-5.0e-324).value(), -5.0e-324)
  assert_eq(@semiring.MinPlus::new(-0.0).value(), 0.0)
}
