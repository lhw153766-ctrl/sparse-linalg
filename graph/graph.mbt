// Graph constructions and algorithms on sparse matrices.
//
// A graph is an adjacency matrix, and for a sparse graph that matrix is sparse:
// a road network with a million junctions has a handful of edges each. This
// package provides the two constructions that come up constantly (an adjacency
// matrix from an edge list, and the Laplacian of one) plus the two algorithms
// that are most often the reason to want a sparse solver in the first place:
// connected components, and PageRank.
//
// The Laplacian matters because it is symmetric positive definite once a
// boundary condition is pinned down, which is exactly what `cg` and sparse
// Cholesky need. PageRank matters because its power iteration is a sparse
// matrix-vector product and nothing else.

///|
/// Degree of every vertex: the number of stored entries in each row, with the
/// diagonal excluded.
pub fn degree(a : @csr.Csr[Double]) -> Array[Int] {
  let n = a.nrows()
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let out : Array[Int] = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    let mut d = 0
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      if ci[k] != i {
        d = d + 1
      }
    }
    out[i] = d
  }
  out
}

///|
/// Weighted degree of every vertex: the sum of edge weights in each row, with
/// the diagonal excluded.
pub fn weighted_degree(a : @csr.Csr[Double]) -> Array[Double] {
  let n = a.nrows()
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let va = a.vals()
  let out : Array[Double] = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    let mut sum = 0.0
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      if ci[k] != i {
        sum = sum + va[k]
      }
    }
    out[i] = sum
  }
  out
}

///|
/// Build a symmetric adjacency matrix from an undirected edge list.
///
/// Each edge is added in both directions. Repeated edges are summed by the
/// coordinate assembly, so a multigraph produces weights rather than an error.
pub fn adjacency_from_edges(
  n : Int,
  edges : Array[(Int, Int)],
) -> @csr.Csr[Double] raise @err.SparseError {
  let coo : @coo.Coo[Double] = @coo.Coo::empty(n, n)
  for e in edges {
    let (u, v) = e
    coo.push(u, v, 1.0)
    if u != v {
      coo.push(v, u, 1.0)
    }
  }
  coo.to_csr()
}

///|
/// Build a symmetric weighted adjacency matrix from an edge list with weights.
pub fn weighted_adjacency_from_edges(
  n : Int,
  edges : Array[(Int, Int, Double)],
) -> @csr.Csr[Double] raise @err.SparseError {
  let coo : @coo.Coo[Double] = @coo.Coo::empty(n, n)
  for e in edges {
    let (u, v, w) = e
    coo.push(u, v, w)
    if u != v {
      coo.push(v, u, w)
    }
  }
  coo.to_csr()
}

///|
/// The combinatorial Laplacian `L = D - A` of a symmetric adjacency matrix.
///
/// Self-loops cancel out of `L = D - A` and are dropped, so every row sums to
/// zero even when the adjacency matrix has entries on its diagonal.
///
/// `L` is symmetric, its rows and columns sum to zero, and it is positive
/// semi-definite with a null space spanned by the constant vector on each
/// connected component. That last fact is why a solve needs a boundary
/// condition: `L x = b` is consistent only when `b` sums to zero.
pub fn laplacian(
  a : @csr.Csr[Double],
) -> @csr.Csr[Double] raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  if !a.is_symmetric() {
    raise @err.SparseError::UnsupportedMatrix(
      "laplacian: the adjacency matrix is not symmetric, so L = D - A would not be symmetric",
    )
  }
  let n = a.nrows()
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let va = a.vals()
  let row_ptr : Array[Int] = Array(capacity=n + 1)
  let col_idx : Array[Int] = Array(capacity=a.nnz())
  let vals : Array[Double] = Array(capacity=a.nnz())
  row_ptr.push(0)
  for i = 0; i < n; i = i + 1 {
    let mut row_sum = 0.0
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      if ci[k] == i {
        // A self-loop cancels out of L = D - A, so the Laplacian of a graph is
        // the same with or without one. Dropping it here is what makes every
        // row sum to zero.
        continue
      }
      col_idx.push(ci[k])
      vals.push(-va[k])
      row_sum = row_sum + va[k]
    }
    // The diagonal of L is the degree, placed in sorted position.
    let mut inserted = false
    let start = row_ptr[i]
    for k = start; k < col_idx.length(); k = k + 1 {
      if !inserted && col_idx[k] > i {
        col_idx.insert(k, i)
        vals.insert(k, row_sum)
        inserted = true
      }
    }
    if !inserted {
      col_idx.push(i)
      vals.push(row_sum)
    }
    row_ptr.push(col_idx.length())
  }
  @csr.Csr::new_unchecked(n, n, row_ptr, col_idx, vals)
}

///|
/// The symmetric normalized Laplacian `L = S (D - A0) S`, where `A0` is the
/// adjacency matrix with self-loops removed.
///
/// For vertices with positive degree, `S_ii = 1 / sqrt(d_i)`, yielding diagonal 1.
/// Self-loops are ignored in degree computation and off-diagonal entries.
/// When `isolated_zero` is true (the default matching SciPy and NetworkX), isolated
/// vertices (`d_i == 0`) have diagonal 0 so that zero eigenvalue multiplicity equals
/// the number of connected components for graphs with positive edge weights.
/// Set `isolated_zero = false` to assign 1 to all diagonals.
pub fn normalized_laplacian(
  a : @csr.Csr[Double],
  isolated_zero? : Bool = true,
) -> @csr.Csr[Double] raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  if !a.is_symmetric() {
    raise @err.SparseError::UnsupportedMatrix(
      "normalized_laplacian: the adjacency matrix is not symmetric",
    )
  }
  let va = a.vals()
  for k = 0; k < va.length(); k = k + 1 {
    let w = va[k]
    if !(w >= 0.0 && w <= 1.7976931348623157e308) {
      raise @err.SparseError::UnsupportedMatrix(
        "normalized_laplacian: edge weights must be finite and non-negative, got " +
        w.to_string(),
      )
    }
  }
  let n = a.nrows()
  let d = weighted_degree(a)
  for i = 0; i < n; i = i + 1 {
    if !(d[i] >= 0.0 && d[i] <= 1.7976931348623157e308) {
      raise @err.SparseError::UnsupportedMatrix(
        "normalized_laplacian: degree of vertex " +
        i.to_string() +
        " overflows floating-point range",
      )
    }
  }
  let sqrt_d : Array[Double] = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    if d[i] > 0.0 {
      sqrt_d[i] = Double::sqrt(d[i])
    }
  }
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let row_ptr : Array[Int] = Array(capacity=n + 1)
  let col_idx : Array[Int] = Array(capacity=a.nnz() + n)
  let vals : Array[Double] = Array(capacity=a.nnz() + n)
  row_ptr.push(0)
  for i = 0; i < n; i = i + 1 {
    if d[i] == 0.0 && isolated_zero {
      row_ptr.push(col_idx.length())
      continue
    }
    let mut inserted = false
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      let j = ci[k]
      if j == i {
        continue
      }
      let w = va[k]
      if w == 0.0 {
        continue
      }
      if !inserted && j > i {
        col_idx.push(i)
        vals.push(1.0)
        inserted = true
      }
      let sqrt_w = Double::sqrt(w)
      let s_i = sqrt_w / sqrt_d[i]
      let s_j = sqrt_w / sqrt_d[j]
      let val = if i <= j { -(s_i * s_j) } else { -(s_j * s_i) }
      col_idx.push(j)
      vals.push(val)
    }
    if !inserted {
      col_idx.push(i)
      vals.push(1.0)
    }
    row_ptr.push(col_idx.length())
  }
  @csr.Csr::new_unchecked(n, n, row_ptr, col_idx, vals)
}

///|
/// Connected component label per vertex, from a symmetric adjacency matrix.
///
/// Labels are assigned in the order components are discovered, starting at 0.
/// Only edges with strictly positive weight connect vertices.
pub fn connected_components(a : @csr.Csr[Double]) -> Array[Int] {
  let n = a.nrows()
  let labels : Array[Int] = Array::make(n, -1)
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let va = a.vals()
  let mut component = 0
  for start = 0; start < n; start = start + 1 {
    if labels[start] >= 0 {
      continue
    }
    // Breadth-first from an unvisited vertex, which is all it takes: every
    // vertex reachable from `start` gets the same label.
    labels[start] = component
    let queue : Array[Int] = Array(capacity=n)
    queue.push(start)
    let mut head = 0
    while head < queue.length() {
      let u = queue[head]
      head = head + 1
      for k = rp[u]; k < rp[u + 1]; k = k + 1 {
        let v = ci[k]
        if v != u && va[k] > 0.0 && labels[v] < 0 {
          labels[v] = component
          queue.push(v)
        }
      }
    }
    component = component + 1
  }
  labels
}

///|
/// Number of connected components.
pub fn component_count(a : @csr.Csr[Double]) -> Int {
  let labels = connected_components(a)
  let mut best = -1
  for l in labels {
    if l > best {
      best = l
    }
  }
  best + 1
}

///|
/// True when every vertex is reachable from every other.
pub fn is_connected(a : @csr.Csr[Double]) -> Bool {
  a.nrows() > 0 && component_count(a) == 1
}

///|
/// PageRank by power iteration.
///
/// `a` is the link matrix: a stored entry `(i, j)` means vertex `i` links to
/// vertex `j`. The returned vector sums to 1.
///
/// Two details that a naive implementation gets wrong. Vertices with no
/// outgoing links — dangling nodes — would leak rank, so their rank is spread
/// over every vertex instead. And the iteration is normalised at every step
/// rather than only at the end, so the convergence test compares comparable
/// quantities.
pub fn page_rank(
  a : @csr.Csr[Double],
  damping? : Double = 0.85,
  tol? : Double = 1.0e-12,
  max_iter? : Int = 200,
) -> Array[Double] raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  if damping.is_nan() || damping.is_inf() || !(damping > 0.0 && damping < 1.0) {
    raise @err.SparseError::DimensionMismatch(
      "page_rank: damping must lie strictly between 0 and 1, got " +
      damping.to_string(),
    )
  }
  if tol.is_nan() || tol.is_inf() || !(tol > 0.0 && tol <= 1.0) {
    raise @err.SparseError::DimensionMismatch(
      "page_rank: tol must be positive and finite, got " + tol.to_string(),
    )
  }
  if max_iter < 1 {
    raise @err.SparseError::DimensionMismatch(
      "page_rank: max_iter must be positive, got " + max_iter.to_string(),
    )
  }
  let n = a.nrows()
  if n == 0 {
    return []
  }
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let va = a.vals()
  let mut has_positive_weight = false
  for k = 0; k < va.length(); k = k + 1 {
    let w = va[k]
    if w.is_nan() || w.is_inf() || w < 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "page_rank: edge weights must be finite and non-negative, got " +
        w.to_string(),
      )
    }
    if w > 0.0 {
      has_positive_weight = true
    }
  }
  if va.length() > 0 && !has_positive_weight {
    raise @err.SparseError::UnsupportedMatrix(
      "page_rank: graph contains edges but no strictly positive weights",
    )
  }
  // Transition probabilities P_ij = w_ij / sum_k(w_ik).
  // Computed using row-max scaling to guarantee scale invariance and prevent
  // intermediate overflow (sum > 1e308) or underflow (subnormal weights).
  let is_dangling : Array[Bool] = Array::make(n, false)
  let probs : Array[Double] = Array::make(va.length(), 0.0)
  for i = 0; i < n; i = i + 1 {
    let mut row_max = 0.0
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      let w = va[k]
      if w > row_max {
        row_max = w
      }
    }
    if row_max == 0.0 {
      is_dangling[i] = true
    } else {
      let mut scaled_sum = 0.0
      for k = rp[i]; k < rp[i + 1]; k = k + 1 {
        scaled_sum = scaled_sum + va[k] / row_max
      }
      if scaled_sum.is_nan() || scaled_sum.is_inf() || scaled_sum <= 0.0 {
        raise @err.SparseError::UnsupportedMatrix(
          "page_rank: row weight sum is non-finite or non-positive at vertex " +
          i.to_string(),
        )
      }
      for k = rp[i]; k < rp[i + 1]; k = k + 1 {
        probs[k] = va[k] / row_max / scaled_sum
      }
    }
  }
  let base = (1.0 - damping) / n.to_double()
  let mut rank : Array[Double] = Array::make(n, 1.0 / n.to_double())
  let mut next : Array[Double] = Array::make(n, 0.0)
  for _iter = 1; _iter <= max_iter; _iter = _iter + 1 {
    // Rank held by dangling vertices is redistributed uniformly.
    let mut dangling = 0.0
    for i = 0; i < n; i = i + 1 {
      if is_dangling[i] {
        dangling = dangling + rank[i]
      }
    }
    let share = damping * dangling / n.to_double()
    for i = 0; i < n; i = i + 1 {
      next[i] = base + share
    }
    // Push each vertex's rank along its outgoing links using precomputed
    // normalized transition probabilities.
    for i = 0; i < n; i = i + 1 {
      if is_dangling[i] {
        continue
      }
      let c_i = damping * rank[i]
      for k = rp[i]; k < rp[i + 1]; k = k + 1 {
        next[ci[k]] = next[ci[k]] + c_i * probs[k]
      }
    }
    // Renormalise first to form a valid probability distribution, then measure
    // the L1 change against the previous iterate.
    let mut total = 0.0
    for i = 0; i < n; i = i + 1 {
      total = total + next[i]
    }
    if total <= 0.0 || total.is_nan() || total.is_inf() {
      raise @err.SparseError::UnsupportedMatrix(
        "page_rank: numerical breakdown: rank distribution sum is non-finite or non-positive",
      )
    }
    let mut delta = 0.0
    for i = 0; i < n; i = i + 1 {
      next[i] = next[i] / total
      delta = delta + Double::abs(next[i] - rank[i])
    }
    let tmp = rank
    rank = next
    next = tmp
    if delta <= tol {
      break
    }
  }
  // Validate final probability distribution contract.
  let mut final_sum = 0.0
  for i = 0; i < n; i = i + 1 {
    let v = rank[i]
    if v.is_nan() || v.is_inf() || v < 0.0 {
      raise @err.SparseError::UnsupportedMatrix(
        "page_rank: final rank vector contains invalid or non-finite values",
      )
    }
    final_sum = final_sum + v
  }
  if Double::abs(final_sum - 1.0) > 1.0e-7 {
    raise @err.SparseError::UnsupportedMatrix(
      "page_rank: final rank distribution failed to normalize to 1.0",
    )
  }
  rank
}

///|
/// A single Jacobi sweep applied repeatedly: `x <- x - omega * D^-1 (A x - b)`.
///
/// Included because it is the simplest thing that works on a Laplacian system
/// and because it shows what a preconditioner is approximating. It converges
/// only for diagonally dominant matrices, and slowly; the Krylov solvers in the
/// `solve` package are the ones to reach for in practice.
pub fn jacobi_sweeps(
  a : @csr.Csr[Double],
  b : Array[Double],
  omega? : Double = 2.0 / 3.0,
  sweeps? : Int = 100,
) -> Array[Double] raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  let n = a.nrows()
  if b.length() != n {
    raise @err.SparseError::DimensionMismatch(
      "jacobi_sweeps: b has length " +
      b.length().to_string() +
      ", expected " +
      n.to_string(),
    )
  }
  let d = a.diagonal()
  for i = 0; i < n; i = i + 1 {
    if d[i] == 0.0 {
      raise @err.SparseError::SingularPivot(i, d[i])
    }
  }
  let x : Array[Double] = Array::make(n, 0.0)
  for _s = 1; _s <= sweeps; _s = _s + 1 {
    let residual = @solve.sub_vec(b, a.spmv(x))
    for i = 0; i < n; i = i + 1 {
      x[i] = x[i] + omega * residual[i] / d[i]
    }
  }
  x
}
