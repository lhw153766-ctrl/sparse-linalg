///|
test "r4 Cholesky tiny positive pivot must not return nonfinite" {
  let a = @ops.identity(1, 1.0e-320)
  let r = Some(@solve.solve_cholesky(a, [1.0])) catch { _ => None }
  if r is Some(x) {
    println("tiny Cholesky x=\{x[0]}")
    assert_true(!x[0].is_nan() && !x[0].is_inf())
  }
}
