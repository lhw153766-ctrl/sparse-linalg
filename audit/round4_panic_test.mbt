///|
test "r4 from_dense wrapped size must raise rather than panic" {
  assert_true(
    r4_rejected(() => {
      let _ = @ops.from_dense(([] : Array[Double]), 65536, 65536, x => x != 0.0)
    }),
  )
}

///|
test "r4 identity maximum size must raise rather than panic" {
  assert_true(
    r4_rejected(() => {
      let _ = @ops.identity(2147483647, 1.0)
    }),
  )
}

///|
test "r4 to_dense size must not silently wrap to empty" {
  let a : @csr.Csr[Double] = @csr.Csr::new(
    65536,
    65536,
    Array::make(65537, 0),
    [],
    [],
  )
  assert_true(
    r4_rejected(() => {
      let _ = a.to_dense()
    }),
  )
}
