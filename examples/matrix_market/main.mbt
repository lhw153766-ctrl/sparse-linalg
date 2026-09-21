// Reading a Matrix Market file, then solving with it.
//
// Matrix Market is the exchange format of the sparse matrix community: the NIST
// and SuiteSparse collections publish in it, so it is how real sparse matrices
// arrive. The document below is in the shape those files have — a banner,
// comment lines naming the source, then the size line and the entries.
//
// Two properties of the format are easy to get wrong and are visible here:
// indices are 1-based, and a symmetric file stores only one triangle, so the
// reader has to mirror it.
//
// Run from the project root:
//
//   moon run examples/matrix_market

///|
/// A symmetric matrix in Matrix Market coordinate form, with the comment block
/// that a collection file carries.
const DOCUMENT : String = "%%MatrixMarket matrix coordinate real symmetric\n" +
  "%-------------------------------------------------------------------------------\n" +
  "% UF Sparse Matrix Collection, Tim Davis\n" +
  "% name: demo/laplace5\n" +
  "% a 5-point stencil on a 3x3 grid, which is what a finite difference\n" +
  "% discretisation of the Laplacian produces\n" +
  "%-------------------------------------------------------------------------------\n" +
  "9 9 21\n" +
  "1 1 4.0\n" +
  "2 1 -1.0\n" +
  "2 2 4.0\n" +
  "3 2 -1.0\n" +
  "3 3 4.0\n" +
  "4 1 -1.0\n" +
  "4 4 4.0\n" +
  "5 2 -1.0\n" +
  "5 4 -1.0\n" +
  "5 5 4.0\n" +
  "6 3 -1.0\n" +
  "6 5 -1.0\n" +
  "6 6 4.0\n" +
  "7 4 -1.0\n" +
  "7 7 4.0\n" +
  "8 5 -1.0\n" +
  "8 7 -1.0\n" +
  "8 8 4.0\n" +
  "9 6 -1.0\n" +
  "9 8 -1.0\n" +
  "9 9 4.0\n"

///|
fn main raise @err.SparseError {
  let file = @io.parse_matrix_market_full(DOCUMENT)
  println("header")
  println("  object    : \{file.header.object}")
  println("  format    : \{file.header.format}")
  println("  field     : \{file.header.field}")
  println("  symmetry  : \{file.header.symmetry}")
  println("")
  println(
    "size line as written: \{file.declared_rows} x \{file.declared_cols}, \{file.declared_nnz} entries",
  )
  println("stored triples after mirroring the triangle: \{file.matrix.nnz()}")
  println("")

  let a = file.matrix.to_csr()
  println("as a matrix")
  println("  shape     : \{a.nrows()} x \{a.ncols()}")
  println("  entries   : \{a.nnz()}")
  println("  symmetric : \{a.is_symmetric()}")
  // A 5-point stencil with 4 on the diagonal has zero row sums at interior
  // points but not at the boundary: a corner has only two neighbours, so its
  // row sums to 2. This is the free-boundary (Neumann) stencil, and it is
  // positive definite, unlike the Laplacian of the graph which has the
  // constant vector in its null space.
  let ones : Array[Double] = Array::make(a.nrows(), 1.0)
  let product = a.spmv(ones)
  let mut worst = 0.0
  for v in product {
    let d = Double::abs(v)
    if d > worst {
      worst = d
    }
  }
  println("  max |A * ones| : \{worst}   (2 at the corners, 0 at the centre)")
  println("")

  let b : Array[Double] = Array::make(9, 0.0)
  b[0] = 1.0
  let result = @solve.cg(a, b)
  println("solving")
  println("  converged         : \{result.converged}")
  println("  iterations        : \{result.iterations}")
  println("  relative residual : \{result.residual}")
  let shown = StringBuilder::StringBuilder()
  for v in result.x {
    shown.write_string("\{v} ")
  }
  println("  solution          : \{shown.to_string()}")
  println("")

  // Writing it back out and reading it again must give the same matrix.
  let text = @io.write_matrix_market(@coo.Coo::from_csr(a))
  let again = @io.parse_matrix_market_csr(text)
  let same = again.to_dense() == a.to_dense()
  println("round trip through the writer gives the same matrix: \{same}")
}
