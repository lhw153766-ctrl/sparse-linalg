// Black-box tests for Matrix Market I/O.
//
// The format has two traps that both produce a matrix that parses cleanly and
// is wrong: indices are 1-based, and a symmetric file stores only one triangle.
// Both directions are tested against each other so a mistake in the reader and
// a matching mistake in the writer cannot cancel out.

///|
fn raises(parse : () -> Unit raise @err.SparseError) -> Bool {
  try {
    parse()
    false
  } catch {
    _ => true
  }
}

///|
test "header parsing" {
  let h = @io.parse_header("%%MatrixMarket matrix coordinate real general")
  inspect(h.object, content="matrix")
  inspect(h.format, content="coordinate")
  inspect(h.field, content="real")
  inspect(h.symmetry, content="general")
  // Case is not significant in the banner.
  let upper = @io.parse_header(
    "%%MatrixMarket MATRIX COORDINATE INTEGER SYMMETRIC",
  )
  inspect(upper.field, content="integer")
  inspect(upper.symmetry, content="symmetric")
  // Extra whitespace is tolerated.
  let padded = @io.parse_header(
    "  %%MatrixMarket   matrix  coordinate   pattern   general  ",
  )
  inspect(padded.field, content="pattern")
}

///|
test "header parsing rejects malformed banners" {
  inspect(
    raises(() => {
      let _ = @io.parse_header("not a banner")
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = @io.parse_header("%%MatrixMarket matrix coordinate real")
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = @io.parse_header("%%MatrixMarket tensor coordinate real general")
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = @io.parse_header("%%MatrixMarket matrix grid real general")
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = @io.parse_header(
        "%%MatrixMarket matrix coordinate quaternion general",
      )
    }),
    content="true",
  )
}

///|
test "read the header without reading the entries" {
  let text = "%%MatrixMarket matrix coordinate real symmetric\n% a comment\n3 3 2\n1 1 1.0\n2 2 2.0\n"
  let h = @io.read_matrix_market_header(text)
  inspect(h.symmetry, content="symmetric")
  inspect(
    raises(() => {
      let _ = @io.read_matrix_market_header("")
    }),
    content="true",
  )
}

///|
test "parse a general coordinate matrix" {
  let text = "%%MatrixMarket matrix coordinate real general\n% a comment\n\n3 3 3\n1 1 2.0\n2 2 3.0\n3 1 -1.5\n"
  let file = @io.parse_matrix_market_full(text)
  inspect(file.declared_rows, content="3")
  inspect(file.declared_cols, content="3")
  inspect(file.declared_nnz, content="3")
  inspect(file.matrix.nnz(), content="3")
  let csr = file.matrix.to_csr()
  // 1-based indices in the file become 0-based: (1,1) is (0,0), (3,1) is (2,0).
  inspect(csr.get_or(0, 0, 0.0), content="2")
  inspect(csr.get_or(1, 1, 0.0), content="3")
  inspect(csr.get_or(2, 0, 0.0), content="-1.5")
  inspect(csr.contains(0, 1), content="false")
}

///|
test "parse a symmetric matrix mirrors the stored triangle" {
  // Only the lower triangle is stored, so (1,2) must appear as well.
  let text = "%%MatrixMarket matrix coordinate real symmetric\n2 2 3\n1 1 4.0\n2 1 1.0\n2 2 3.0\n"
  let a = @io.parse_matrix_market(text)
  inspect(a.nnz(), content="4")
  let csr = a.to_csr()
  debug_inspect(csr.to_dense(), content="[4, 1, 1, 3]")
  inspect(csr.is_symmetric(), content="true")
}

///|
test "parse a skew-symmetric matrix negates the mirror" {
  let text = "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 1\n2 1 5.0\n"
  let a = @io.parse_matrix_market(text)
  inspect(a.nnz(), content="2")
  let csr = a.to_csr()
  debug_inspect(csr.to_dense(), content="[0, -5, 5, 0]")
}

///|
test "a diagonal entry in a skew-symmetric file is not mirrored twice" {
  let text = "%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 2\n1 1 0.0\n2 1 3.0\n"
  let a = @io.parse_matrix_market(text)
  // (1,1) is diagonal so it is stored once; (2,1) produces both (1,0) and (0,1).
  inspect(a.nnz(), content="3")
}

///|
test "parse a pattern matrix gives every entry the value one" {
  let text = "%%MatrixMarket matrix coordinate pattern general\n2 3 2\n1 1\n2 3\n"
  let a = @io.parse_matrix_market(text)
  inspect(a.nnz(), content="2")
  let csr = a.to_csr()
  inspect(csr.get_or(0, 0, 0.0), content="1")
  inspect(csr.get_or(1, 2, 0.0), content="1")
}

///|
test "parse an integer matrix widens the values" {
  let text = "%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 7\n2 2 -3\n"
  let a = @io.parse_matrix_market(text)
  inspect(a.to_csr().get_or(1, 1, 0.0), content="-3")
}

///|
test "parse an array matrix in column-major order" {
  // Array format lists every entry, column by column, with no index columns.
  let text = "%%MatrixMarket matrix array real general\n2 3\n1.0\n2.0\n0.0\n3.0\n4.0\n5.0\n"
  let a = @io.parse_matrix_market(text)
  let csr = a.to_csr()
  inspect(csr.nrows(), content="2")
  inspect(csr.ncols(), content="3")
  // Column 0 is [1, 2], column 1 is [0, 3], column 2 is [4, 5].
  debug_inspect(csr.to_dense(), content="[1, 0, 4, 2, 3, 5]")
  // The explicit zero in the file is not stored.
  inspect(csr.nnz(), content="5")
}

///|
test "tabs and carriage returns are accepted as separators" {
  let text = "%%MatrixMarket matrix coordinate real general\r\n2 2 2\r\n1\t1\t1.5\r\n2\t2\t2.5\r\n"
  let a = @io.parse_matrix_market(text)
  inspect(a.to_csr().get_or(0, 0, 0.0), content="1.5")
  inspect(a.to_csr().get_or(1, 1, 0.0), content="2.5")
}

///|
test "write then read reproduces the matrix" {
  let original : @coo.Coo[Double] = @coo.Coo::new(
    3,
    4,
    [0, 0, 2, 2],
    [0, 3, 1, 3],
    [1.5, -2.0, 3.0, 4.25],
  )
  let text = @io.write_matrix_market(original)
  let back = @io.parse_matrix_market(text)
  inspect(back.nrows(), content="3")
  inspect(back.ncols(), content="4")
  inspect(back.nnz(), content="4")
  let a = original.to_csr()
  let b = back.to_csr()
  for i = 0; i < 3; i = i + 1 {
    for j = 0; j < 4; j = j + 1 {
      inspect(a.get_or(i, j, 0.0), content=b.get_or(i, j, 0.0).to_string())
    }
  }
}

///|
test "write then read round trips a symmetric matrix through the compact form" {
  let full : @coo.Coo[Double] = @coo.Coo::new(2, 2, [0, 1, 1], [0, 0, 1], [
    4.0, 1.0, 3.0,
  ])
  let text = @io.write_matrix_market_symmetric(full)
  // Only the lower triangle is written: 3 of the 4 entries.
  inspect(text.contains("2 2 3"), content="true")
  let back = @io.parse_matrix_market(text)
  let csr = back.to_csr()
  debug_inspect(csr.to_dense(), content="[4, 1, 1, 3]")
  inspect(csr.is_symmetric(), content="true")
}

///|
test "writing a symmetric file rejects a non-square matrix" {
  let rect : @coo.Coo[Double] = @coo.Coo::new(2, 3, [0], [0], [1.0])
  inspect(
    raises(() => {
      let _ = @io.write_matrix_market_symmetric(rect)
    }),
    content="true",
  )
}

///|
test "the written header names the format the writer actually used" {
  let a : @coo.Coo[Double] = @coo.Coo::new(1, 1, [0], [0], [1.0])
  inspect(
    @io.write_matrix_market(a).contains(
      "%%MatrixMarket matrix coordinate real general",
    ),
    content="true",
  )
  inspect(
    @io.write_matrix_market_symmetric(a).contains(
      "%%MatrixMarket matrix coordinate real symmetric",
    ),
    content="true",
  )
}

///|
test "parse into csr directly" {
  let text = "%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1.0\n2 2 2.0\n"
  let csr = @io.parse_matrix_market_csr(text)
  inspect(csr.nnz(), content="2")
  debug_inspect(csr.spmv([1.0, 1.0]), content="[1, 2]")
}

///|
test "a matrix from the classic collection style header" {
  // The NIST files carry several comment lines after the banner and before the
  // size line; they must all be skipped.
  let text = "%%MatrixMarket matrix coordinate real symmetric\n" +
    "%-------------------------------------------------------------------------------\n" +
    "% UF Sparse Matrix Collection, Tim Davis\n" +
    "% name: HB/bcsstk01\n" +
    "%-------------------------------------------------------------------------------\n" +
    "2 2 3\n" +
    "1 1 1.9e6\n" +
    "2 1 -4.0e5\n" +
    "2 2 2.0e6\n"
  let a = @io.parse_matrix_market(text)
  let csr = a.to_csr()
  inspect(csr.is_symmetric(), content="true")
  inspect(csr.get_or(0, 0, 0.0), content="1900000")
  inspect(csr.get_or(1, 0, 0.0), content="-400000")
  inspect(csr.get_or(0, 1, 0.0), content="-400000")
}

///|
test "malformed documents are rejected" {
  // Empty document.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market("")
    }),
    content="true",
  )
  // Header but no size line.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n",
      )
    }),
    content="true",
  )
  // Declares three entries but only two are present.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 1.0\n2 2 2.0\n",
      )
    }),
    content="true",
  )
  // Entry line with a missing value.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1\n",
      )
    }),
    content="true",
  )
  // Non-numeric index.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n2 2 1\nx 1 1.0\n",
      )
    }),
    content="true",
  )
  // Non-numeric value.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 abc\n",
      )
    }),
    content="true",
  )
  // Index out of range: the file is 1-based, so 3 is invalid for a 2x2 matrix.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n2 2 1\n3 1 1.0\n",
      )
    }),
    content="true",
  )
  // Zero index, which would be valid if the file were 0-based.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n2 2 1\n0 1 1.0\n",
      )
    }),
    content="true",
  )
  // Negative dimensions.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n-2 2 0\n",
      )
    }),
    content="true",
  )
  // Size line with too few fields.
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real general\n2 2\n",
      )
    }),
    content="true",
  )
}

///|
test "complex and hermitian inputs are refused rather than silently truncated" {
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate complex general\n2 2 1\n1 1 1.0 2.0\n",
      )
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix coordinate real hermitian\n2 2 1\n1 1 1.0\n",
      )
    }),
    content="true",
  )
}

///|
test "array format cannot claim symmetry" {
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix array real symmetric\n2 2\n1.0\n2.0\n3.0\n4.0\n",
      )
    }),
    content="true",
  )
}

///|
test "array format with too few values is rejected" {
  inspect(
    raises(() => {
      let _ = @io.parse_matrix_market(
        "%%MatrixMarket matrix array real general\n2 3\n1.0\n2.0\n",
      )
    }),
    content="true",
  )
}

///|
test "an empty matrix round trips" {
  let empty : @coo.Coo[Double] = @coo.Coo::empty(3, 3)
  let text = @io.write_matrix_market(empty)
  inspect(text.contains("3 3 0"), content="true")
  let back = @io.parse_matrix_market(text)
  inspect(back.nnz(), content="0")
  inspect(back.nrows(), content="3")
}
