// Matrix Market I/O.
//
// The Matrix Market format is the exchange format of the sparse matrix
// community: the NIST collection and the SuiteSparse collection both publish
// matrices in it, and it is plain text, which makes it useful for test fixtures
// as well as for real data.
//
// The format is a header line, then comment lines, then a size line, then the
// entries:
//
//   %%MatrixMarket matrix coordinate real general
//   % a comment
//   3 3 4
//   1 1 2.0
//   2 2 3.0
//
// Two details matter. Indices are **1-based**, and for a symmetric matrix only
// one triangle is stored — the reader has to mirror it, and the writer has to
// decide whether to. Getting either wrong yields a matrix that parses cleanly
// and is wrong, which is why both directions are tested against each other.
//
// Everything here works on strings rather than files, so the package keeps
// working on every backend, including the ones without a filesystem.

///|
/// The parsed header of a Matrix Market file.
pub(all) struct MatrixMarketHeader {
  /// `matrix` or `vector`.
  object : String
  /// `coordinate` or `array`.
  format : String
  /// `real`, `integer`, `pattern` or `complex`.
  field : String
  /// `general`, `symmetric`, `skew-symmetric` or `hermitian`.
  symmetry : String
} derive(Debug, Eq)

///|
/// A parsed file: the matrix, its header, and the declared size line.
pub(all) struct MatrixMarketFile {
  matrix : @coo.Coo[Double]
  header : MatrixMarketHeader
  /// The `nrows ncols nnz` triple as written in the file, before symmetry was
  /// expanded. Kept so that a round trip can be checked against the original.
  declared_rows : Int
  declared_cols : Int
  declared_nnz : Int
} derive(Debug)

///|
fn parse_error(msg : String) -> @err.SparseError {
  @err.SparseError::ParseError(msg)
}

// Split into non-empty, trimmed lines, keeping every one. The banner starts
// with `%` just like a comment does, so it cannot be filtered out here; the
// caller takes it off the front and only then drops the comments.

///|
fn all_lines(text : String) -> Array[String] {
  let out : Array[String] = Array(capacity=0)
  for raw in text.split("\n") {
    let line = raw.trim()
    if line.length() > 0 {
      out.push(line.to_owned())
    }
  }
  out
}

// Everything after the banner, with comment lines removed. The first element is
// then the size line.

///|
fn body_after_banner(
  lines : Array[String],
) -> Array[String] raise @err.SparseError {
  if lines.length() == 0 {
    raise parse_error("the document has no header line")
  }
  let out : Array[String] = Array(capacity=lines.length())
  for k = 1; k < lines.length(); k = k + 1 {
    if !lines[k].has_prefix("%") {
      out.push(lines[k])
    }
  }
  out
}

///|
fn split_fields(line : String) -> Array[String] {
  let out : Array[String] = Array(capacity=0)
  for f in line.split(" ") {
    let t = f.trim()
    if t.length() > 0 {
      out.push(t.to_owned())
    }
  }
  out
}

// Tabs are legal separators in practice, so normalise them before splitting.

///|
fn normalise_separators(line : String) -> String {
  line.replace_all(old="\t", new=" ").replace_all(old="\r", new="")
}

///|
fn require_int(s : String, what : String) -> Int raise @err.SparseError {
  @string.parse_int(s) catch {
    _ =>
      raise parse_error("expected an integer for " + what + ", got '" + s + "'")
  }
}

///|
fn require_double(s : String, what : String) -> Double raise @err.SparseError {
  @string.parse_double(s) catch {
    _ =>
      raise parse_error("expected a number for " + what + ", got '" + s + "'")
  }
}

///|
/// Parse the `%%MatrixMarket` header line.
pub fn parse_header(line : String) -> MatrixMarketHeader raise @err.SparseError {
  let fields = split_fields(normalise_separators(line))
  if fields.length() < 5 {
    raise parse_error(
      "header needs five fields, got " +
      fields.length().to_string() +
      ": '" +
      line +
      "'",
    )
  }
  if fields[0].to_lower() != "%%matrixmarket" {
    raise parse_error(
      "expected a %%MatrixMarket banner, got '" + fields[0] + "'",
    )
  }
  let object = fields[1].to_lower()
  let format = fields[2].to_lower()
  let field = fields[3].to_lower()
  let symmetry = fields[4].to_lower()
  if object != "matrix" && object != "vector" {
    raise parse_error(
      "unsupported object '" + object + "', expected matrix or vector",
    )
  }
  if format != "coordinate" && format != "array" {
    raise parse_error(
      "unsupported format '" + format + "', expected coordinate or array",
    )
  }
  if field != "real" &&
    field != "integer" &&
    field != "pattern" &&
    field != "complex" {
    raise parse_error("unsupported field '" + field + "'")
  }
  if symmetry != "general" &&
    symmetry != "symmetric" &&
    symmetry != "skew-symmetric" &&
    symmetry != "hermitian" {
    raise parse_error("unsupported symmetry '" + symmetry + "'")
  }
  { object, format, field, symmetry, }
}

///|
/// Read the header of a Matrix Market document without reading the entries.
pub fn read_matrix_market_header(
  text : String,
) -> MatrixMarketHeader raise @err.SparseError {
  for raw in text.split("\n") {
    let line = raw.trim()
    if line.length() > 0 {
      return parse_header(line.to_owned())
    }
  }
  raise parse_error("the document is empty")
}

///|
/// Parse a Matrix Market document into coordinate form.
///
/// Symmetric and skew-symmetric files are expanded, so the returned matrix is
/// always the full one. `pattern` files get the value 1.0, and `integer` files
/// are widened to `Double`.
pub fn parse_matrix_market_full(
  text : String,
) -> MatrixMarketFile raise @err.SparseError {
  let all = all_lines(text)
  if all.length() == 0 {
    raise parse_error("the document has no header line")
  }
  let header = parse_header(normalise_separators(all[0]))
  let lines = body_after_banner(all)
  if header.field == "complex" {
    raise parse_error(
      "complex matrices are not supported: the real part would be silently kept and the imaginary part lost",
    )
  }
  if header.symmetry == "hermitian" {
    raise parse_error(
      "hermitian symmetry is only meaningful for complex matrices, which are not supported",
    )
  }
  if lines.length() < 1 {
    raise parse_error("the document has a header but no size line")
  }
  let size = split_fields(normalise_separators(lines[0]))
  let is_coordinate = header.format == "coordinate"
  let needed = if is_coordinate { 3 } else { 2 }
  if size.length() < needed {
    raise parse_error(
      "the size line needs " +
      needed.to_string() +
      " fields, got " +
      size.length().to_string(),
    )
  }
  let nrows = require_int(size[0], "the row count")
  let ncols = require_int(size[1], "the column count")
  if nrows < 0 || ncols < 0 {
    raise parse_error(
      "negative dimensions " + nrows.to_string() + "x" + ncols.to_string(),
    )
  }
  let declared_nnz = if is_coordinate {
    require_int(size[2], "the entry count")
  } else {
    nrows * ncols
  }
  if declared_nnz < 0 {
    raise parse_error("negative entry count " + declared_nnz.to_string())
  }
  let coo : @coo.Coo[Double] = @coo.Coo::empty(nrows, ncols)
  if is_coordinate {
    let has_value = header.field != "pattern"
    let width = if has_value { 3 } else { 2 }
    for k = 0; k < declared_nnz; k = k + 1 {
      let line_index = 1 + k
      if line_index >= lines.length() {
        raise parse_error(
          "the file declares " +
          declared_nnz.to_string() +
          " entries but only " +
          (lines.length() - 1).to_string() +
          " are present",
        )
      }
      let f = split_fields(normalise_separators(lines[line_index]))
      if f.length() < width {
        raise parse_error(
          "entry " +
          (k + 1).to_string() +
          " needs " +
          width.to_string() +
          " fields: '" +
          lines[line_index] +
          "'",
        )
      }
      let i = require_int(f[0], "a row index") - 1
      let j = require_int(f[1], "a column index") - 1
      let v = if has_value {
        require_double(f[2], "an entry value")
      } else {
        1.0
      }
      add_entry(coo, i, j, v, header.symmetry)
    }
  } else {
    // Array format: every entry of the matrix, in column-major order, and no
    // index columns. Symmetry is not meaningful here.
    if header.symmetry != "general" {
      raise parse_error(
        "array format cannot carry symmetry '" + header.symmetry + "'",
      )
    }
    let has_value = header.field != "pattern"
    let expected = nrows * ncols
    let available = lines.length() - 1
    if available < expected {
      raise parse_error(
        "array format needs " +
        expected.to_string() +
        " values but the file has " +
        available.to_string(),
      )
    }
    for j = 0; j < ncols; j = j + 1 {
      for i = 0; i < nrows; i = i + 1 {
        let line = lines[1 + j * nrows + i]
        let v = if has_value {
          require_double(
            split_fields(normalise_separators(line))[0],
            "an entry value",
          )
        } else {
          1.0
        }
        if v != 0.0 {
          coo.push(i, j, v)
        }
      }
    }
  }
  {
    matrix: coo,
    header,
    declared_rows: nrows,
    declared_cols: ncols,
    declared_nnz,
  }
}

// Add one entry, mirroring it when the file stores only one triangle. The
// mirror of a skew-symmetric entry carries the opposite sign.

///|
fn add_entry(
  coo : @coo.Coo[Double],
  i : Int,
  j : Int,
  v : Double,
  symmetry : String,
) -> Unit raise @err.SparseError {
  if i < 0 || i >= coo.nrows() || j < 0 || j >= coo.ncols() {
    raise @err.SparseError::IndexOutOfBounds(i, j, coo.nrows(), coo.ncols())
  }
  coo.push(i, j, v)
  if i != j {
    if symmetry == "symmetric" {
      coo.push(j, i, v)
    } else if symmetry == "skew-symmetric" {
      coo.push(j, i, -v)
    }
  }
}

///|
/// Parse a Matrix Market document, discarding the header.
pub fn parse_matrix_market(
  text : String,
) -> @coo.Coo[Double] raise @err.SparseError {
  parse_matrix_market_full(text).matrix
}

///|
/// Parse straight into CSR, which is the form the solvers take.
pub fn parse_matrix_market_csr(
  text : String,
) -> @csr.Csr[Double] raise @err.SparseError {
  parse_matrix_market_full(text).matrix.to_csr()
}

// Format a value so that it reads back exactly. Integers lose their trailing
// ".0" because that is what the NIST files do, and `to_string` is already the
// shortest representation that round-trips.

///|
fn format_value(v : Double) -> String {
  v.to_string()
}

///|
/// Write a matrix in coordinate form with `general` symmetry.
///
/// Every stored entry is written, including any that happen to be zero, so the
/// output describes the pattern rather than the numerical non-zeros. Reading it
/// back gives the same matrix.
pub fn write_matrix_market(a : @coo.Coo[Double]) -> String {
  let buf = StringBuilder::StringBuilder()
  buf.write_string("%%MatrixMarket matrix coordinate real general\n")
  buf.write_string("% generated by lhw153766-ctrl/sparse-linalg\n")
  buf.write_string(
    a.nrows().to_string() +
    " " +
    a.ncols().to_string() +
    " " +
    a.nnz().to_string() +
    "\n",
  )
  let rows = a.rows()
  let cols = a.cols()
  let vals = a.vals()
  for k = 0; k < vals.length(); k = k + 1 {
    buf.write_string(
      (rows[k] + 1).to_string() +
      " " +
      (cols[k] + 1).to_string() +
      " " +
      format_value(vals[k]) +
      "\n",
    )
  }
  buf.to_string()
}

///|
/// Write a symmetric matrix storing only its lower triangle.
///
/// Halves the file size for a symmetric matrix. The reader mirrors the entries
/// back, so the round trip still reproduces the full matrix. An entry that has
/// no mirror in the input is still written, which means the file no longer
/// describes a symmetric matrix; that is the caller's choice to make, and
/// `write_matrix_market` is the safe default.
pub fn write_matrix_market_symmetric(
  a : @coo.Coo[Double],
) -> String raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  let rows = a.rows()
  let cols = a.cols()
  let vals = a.vals()
  let kept : Array[Int] = Array(capacity=vals.length())
  for k = 0; k < vals.length(); k = k + 1 {
    if rows[k] >= cols[k] {
      kept.push(k)
    }
  }
  let buf = StringBuilder::StringBuilder()
  buf.write_string("%%MatrixMarket matrix coordinate real symmetric\n")
  buf.write_string("% generated by lhw153766-ctrl/sparse-linalg\n")
  buf.write_string(
    a.nrows().to_string() +
    " " +
    a.ncols().to_string() +
    " " +
    kept.length().to_string() +
    "\n",
  )
  for idx in kept {
    buf.write_string(
      (rows[idx] + 1).to_string() +
      " " +
      (cols[idx] + 1).to_string() +
      " " +
      format_value(vals[idx]) +
      "\n",
    )
  }
  buf.to_string()
}
