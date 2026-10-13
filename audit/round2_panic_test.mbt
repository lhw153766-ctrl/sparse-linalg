///|
test "r2 coordinate maximum row dimension CSR pointer overflow" {
  assert_true(
    rejected(() => {
      let _ = @io.parse_matrix_market_csr(
        "%%MatrixMarket matrix coordinate real general\n2147483647 0 0\n",
      )
    }),
  )
}
