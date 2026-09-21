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
/// Total number of entries a dense matrix of this shape would hold.
pub fn Shape::dense_len(self : Shape) -> Int {
  self.rows * self.cols
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
  let dense = shape.dense_len()
  if dense == 0 {
    return 0.0
  }
  nnz.to_double() / dense.to_double()
}
