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
/// The normalized Laplacian `I - D^(-1/2) A D^(-1/2)`.
///
/// Its eigenvalues lie in `[0, 2]`, which makes it the usual choice for
/// spectral partitioning.
pub fn normalized_laplacian(
  a : @csr.Csr[Double],
) -> @csr.Csr[Double] raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  let n = a.nrows()
  let d = degree(a)
  let inv_sqrt : Array[Double] = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    if d[i] > 0 {
      inv_sqrt[i] = 1.0 / Double::sqrt(d[i].to_double())
    }
  }
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let va = a.vals()
  let row_ptr : Array[Int] = Array(capacity=n + 1)
  let col_idx : Array[Int] = Array(capacity=a.nnz())
  let vals : Array[Double] = Array(capacity=a.nnz())
  row_ptr.push(0)
  for i = 0; i < n; i = i + 1 {
    let mut inserted = false
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      let j = ci[k]
      if j == i {
        continue
      }
      if !inserted && j > i {
        col_idx.push(i)
        vals.push(1.0)
        inserted = true
      }
      col_idx.push(j)
      vals.push(-va[k] * inv_sqrt[i] * inv_sqrt[j])
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
pub fn connected_components(a : @csr.Csr[Double]) -> Array[Int] {
  let n = a.nrows()
  let labels : Array[Int] = Array::make(n, -1)
  let rp = a.row_ptr()
  let ci = a.col_idx()
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
        if labels[v] < 0 {
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
  if !(damping > 0.0 && damping < 1.0) {
    raise @err.SparseError::DimensionMismatch(
      "page_rank: damping must lie strictly between 0 and 1, got " +
      damping.to_string(),
    )
  }
  let n = a.nrows()
  if n == 0 {
    return []
  }
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let va = a.vals()
  // Out-weight of each vertex, and whether it is dangling.
  let out_weight : Array[Double] = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    let mut s = 0.0
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      s = s + va[k]
    }
    out_weight[i] = s
  }
  let base = (1.0 - damping) / n.to_double()
  let mut rank : Array[Double] = Array::make(n, 1.0 / n.to_double())
  let mut next : Array[Double] = Array::make(n, 0.0)
  for _iter = 1; _iter <= max_iter; _iter = _iter + 1 {
    // Rank held by dangling vertices is redistributed uniformly.
    let mut dangling = 0.0
    for i = 0; i < n; i = i + 1 {
      if out_weight[i] == 0.0 {
        dangling = dangling + rank[i]
      }
    }
    let share = damping * dangling / n.to_double()
    for i = 0; i < n; i = i + 1 {
      next[i] = base + share
    }
    // Push each vertex's rank along its outgoing links.
    for i = 0; i < n; i = i + 1 {
      let w = out_weight[i]
      if w == 0.0 {
        continue
      }
      let contribution = damping * rank[i] / w
      for k = rp[i]; k < rp[i + 1]; k = k + 1 {
        next[ci[k]] = next[ci[k]] + contribution * va[k]
      }
    }
    // Convergence is measured on the L1 change, which is the standard choice
    // because the vector is a probability distribution.
    let mut delta = 0.0
    let mut total = 0.0
    for i = 0; i < n; i = i + 1 {
      delta = delta + Double::abs(next[i] - rank[i])
      total = total + next[i]
    }
    // Renormalise to absorb rounding, then swap the buffers.
    if total > 0.0 {
      for i = 0; i < n; i = i + 1 {
        next[i] = next[i] / total
      }
    }
    let tmp = rank
    rank = next
    next = tmp
    if delta <= tol {
      return rank
    }
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
