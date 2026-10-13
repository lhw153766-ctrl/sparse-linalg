///|
test "r3 maximum empty COO conversion must not abort" {
  assert_true(
    rejected(() => {
      let a : @coo.Coo[Double] = @coo.Coo::empty(2147483647, 0)
      let _ = a.to_csr()
    }),
  )
}
