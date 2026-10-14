// The variants are deliberately specific: a caller that feeds a non-symmetric
// matrix to a conjugate-gradient solver should be able to tell that apart from
// a genuine lack of convergence, and a caller parsing a Matrix Market file
// should be able to tell a malformed header from an out-of-range index.

///|
/// Errors raised while building, combining, or solving sparse systems.
pub(all) suberror SparseError {
  /// An index was negative or outside the declared shape.
  /// Carries `(row, col, nrows, ncols)` so the caller can report both.
  IndexOutOfBounds(Int, Int, Int, Int)
  /// Two operands did not have compatible shapes for the requested operation.
  /// The string names the operation that failed.
  DimensionMismatch(String)
  /// The operation needs a square matrix; carries the actual `(nrows, ncols)`.
  NotSquare(Int, Int)
  /// A factorisation met a pivot that is not strictly positive, so the matrix
  /// is not positive definite. Carries the step index and the offending value.
  NotPositiveDefinite(Int, Double)
  /// A factorisation met a pivot that is exactly zero.
  SingularPivot(Int, Double)
  /// An iterative solver used its whole iteration budget without reaching the
  /// requested tolerance. Carries `(iterations_used, achieved_residual)`.
  NotConverged(Int, Double)
  /// A solver was handed a matrix that violates its assumptions, such as a
  /// structurally asymmetric matrix given to a symmetric-only solver.
  UnsupportedMatrix(String)
  /// Textual input (Matrix Market, coordinate lists) could not be parsed.
  ParseError(String)
} derive(Debug, Eq)

///|
/// The shape of a matrix, in row-major order: `rows` by `cols`.
pub(all) struct Shape {
  rows : Int
  cols : Int
} derive(Debug, Eq)

///|
/// Construct a `Shape`, rejecting negative extents.
pub fn Shape::new(rows : Int, cols : Int) -> Shape raise SparseError {
  if rows < 0 || cols < 0 {
    raise SparseError::DimensionMismatch(
      "Shape::new: negative extent " + rows.to_string() + "x" + cols.to_string(),
    )
  }
  { rows, cols, }
}

///|
/// Number of rows.
pub fn Shape::rows(self : Shape) -> Int {
  self.rows
}

///|
/// Number of columns.
pub fn Shape::cols(self : Shape) -> Int {
  self.cols
}

///|
/// Resource policy limit for dense entries. This bounds requested allocation
/// sizes; it cannot guarantee available memory in every host process.
pub const MAX_DENSE_ELEMENTS : Int64 = 30_000_000L

///|
/// Resource policy limit for each sparse matrix dimension.
pub const MAX_SPARSE_DIMENSION : Int = 1_000_000

///|
/// Safe upper bound on sparse non-zero entry count across backends.
pub const MAX_SPARSE_NNZ : Int = 10_000_000

///|
/// Validate dimensions independently of their product, including zero shapes.
pub fn check_sparse_shape(nrows : Int, ncols : Int) -> Unit raise SparseError {
  if nrows < 0 ||
    ncols < 0 ||
    nrows > MAX_SPARSE_DIMENSION ||
    ncols > MAX_SPARSE_DIMENSION {
    raise SparseError::DimensionMismatch(
      "sparse dimensions exceed supported bounds: " +
      nrows.to_string() +
      "x" +
      ncols.to_string(),
    )
  }
}

///|
/// Check entry counts before allocation or growth; narrow only after checking.
pub fn check_sparse_nnz(count : Int64) -> Int raise SparseError {
  if count < 0L || count > MAX_SPARSE_NNZ.to_int64() {
    raise SparseError::DimensionMismatch(
      "sparse entry capacity exceeded: " + count.to_string(),
    )
  }
  count.to_int()
}

///|
/// Bound the total scalar slots requested by an algorithm's work arrays.
pub fn check_workspace_capacity(count : Int64) -> Unit raise SparseError {
  if count < 0L || count > MAX_DENSE_ELEMENTS {
    raise SparseError::DimensionMismatch(
      "workspace capacity exceeded: " + count.to_string(),
    )
  }
}

///|
/// Check that `nrows` and `ncols` can be represented as a dense array,
/// rejecting negative dimensions, integer overflow, and arrays exceeding
/// the configured dense capacity (30,000,000 elements).
pub fn check_dense_capacity(nrows : Int, ncols : Int) -> Int raise SparseError {
  check_sparse_shape(nrows, ncols)
  let r = nrows.to_int64()
  let c = ncols.to_int64()
  let total = r * c
  if total < 0L || total > MAX_DENSE_ELEMENTS {
    raise SparseError::DimensionMismatch(
      "dense capacity exceeded or overflow: requested " +
      nrows.to_string() +
      "x" +
      ncols.to_string() +
      " (" +
      total.to_string() +
      " elements), max supported is " +
      MAX_DENSE_ELEMENTS.to_string(),
    )
  }
  total.to_int()
}

///|
/// Total number of entries a dense matrix of this shape would hold.
pub fn Shape::dense_len(self : Shape) -> Int raise SparseError {
  check_dense_capacity(self.rows, self.cols)
}

///|
/// True when the shape has the same number of rows and columns.
pub fn Shape::is_square(self : Shape) -> Bool {
  self.rows == self.cols
}

///|
/// Check that `(row, col)` addresses an entry inside `shape`.
pub fn Shape::check_index(
  self : Shape,
  row : Int,
  col : Int,
) -> Unit raise SparseError {
  if row < 0 || row >= self.rows || col < 0 || col >= self.cols {
    raise SparseError::IndexOutOfBounds(row, col, self.rows, self.cols)
  }
}

///|
/// The count of stored entries relative to the dense size of the same shape.
///
/// A value of 1.0 means every entry is stored; small values mean the sparse
/// representation is winning. Returned as `Double` because callers use it for
/// reporting and for choosing between dense and sparse algorithms.
pub fn density(shape : Shape, nnz : Int) -> Double {
  let r = shape.rows.to_int64()
  let c = shape.cols.to_int64()
  let total = r * c
  if total <= 0L {
    0.0
  } else {
    nnz.to_double() / total.to_double()
  }
}
