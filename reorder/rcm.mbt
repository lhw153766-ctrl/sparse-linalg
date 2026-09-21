// Bandwidth-reducing reordering.
//
// A sparse direct factorisation is only as sparse as the matrix lets it be.
// Eliminating an unknown couples every pair of its neighbours, so a matrix
// whose non-zeros are spread far from the diagonal fills in heavily, while one
// whose non-zeros hug the diagonal barely fills at all.
//
// The reverse Cuthill-McKee ordering reduces the bandwidth: it numbers the
// vertices in breadth-first order, always expanding the lowest-degree neighbour
// first, then reverses the result. Reversing is not cosmetic — it is what makes
// the ordering *reverse* Cuthill-McKee, and it produces a profile at least as
// good as the unreversed order in practice.
//
// This is a heuristic, and a cheap one. Minimum-degree orderings and nested
// dissection generally beat it; they are also far more code.

///|
/// Maximum `|i - j|` over the stored entries.
///
/// This is the quantity RCM is trying to reduce, and the one to measure before
/// and after to see whether a reordering helped.
pub fn bandwidth(a : @csr.Csr[Double]) -> Int {
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let mut worst = 0
  for i = 0; i < a.nrows(); i = i + 1 {
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      let d = if ci[k] > i { ci[k] - i } else { i - ci[k] }
      if d > worst {
        worst = d
      }
    }
  }
  worst
}

///|
/// The profile: the sum over rows of the distance from the row index to the
/// leftmost stored column.
///
/// A better predictor of factorisation cost than the bandwidth, because it
/// measures the total work rather than the worst single row.
pub fn profile(a : @csr.Csr[Double]) -> Int {
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let mut total = 0
  for i = 0; i < a.nrows(); i = i + 1 {
    if rp[i] < rp[i + 1] {
      let first = ci[rp[i]]
      if i > first {
        total = total + (i - first)
      }
    }
  }
  total
}

///|
/// The inverse of a permutation: `inv[perm[i]] = i`.
pub fn invert_permutation(
  perm : Array[Int],
) -> Array[Int] raise @err.SparseError {
  let n = perm.length()
  let inv : Array[Int] = Array::make(n, -1)
  for i = 0; i < n; i = i + 1 {
    let p = perm[i]
    if p < 0 || p >= n {
      raise @err.SparseError::IndexOutOfBounds(p, i, n, n)
    }
    if inv[p] != -1 {
      raise @err.SparseError::DimensionMismatch(
        "invert_permutation: " +
        p.to_string() +
        " appears twice, so this is not a permutation",
      )
    }
    inv[p] = i
  }
  inv
}

///|
/// True when the array is a permutation of `0..n-1`.
pub fn is_permutation(perm : Array[Int], n : Int) -> Bool {
  if perm.length() != n {
    return false
  }
  let seen : Array[Bool] = Array::make(n, false)
  for p in perm {
    if p < 0 || p >= n || seen[p] {
      return false
    }
    seen[p] = true
  }
  true
}

///|
/// Apply a permutation, using the convention `perm[old_index] = new_index`.
///
/// That is, the entry at `(i, j)` moves to `(perm[i], perm[j])`. The same
/// convention is used by `reverse_cuthill_mckee`, so
/// `permute(a, reverse_cuthill_mckee(a))` is the reordered matrix.
///
/// The pattern is preserved exactly — this is a relabelling, not an
/// approximation — so the reordered matrix has the same number of stored
/// entries and the same eigenvalues. Only the factorisation cost changes.
pub fn permute(
  a : @csr.Csr[Double],
  perm : Array[Int],
) -> @csr.Csr[Double] raise @err.SparseError {
  let n = a.nrows()
  if !is_permutation(perm, n) {
    raise @err.SparseError::DimensionMismatch(
      "permute: expected a permutation of 0.." + (n - 1).to_string(),
    )
  }
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let va = a.vals()
  let coo : @coo.Coo[Double] = @coo.Coo::empty(n, n)
  for i = 0; i < n; i = i + 1 {
    for k = rp[i]; k < rp[i + 1]; k = k + 1 {
      // Entry (i, ci[k]) moves to (perm[i], perm[ci[k]]).
      coo.push(perm[i], perm[ci[k]], va[k])
    }
  }
  coo.to_csr()
}

///|
/// Reverse Cuthill-McKee ordering of a symmetric matrix.
///
/// Returns `perm` such that `permute(a, perm)` has reduced bandwidth, using the
/// convention `perm[old_index] = new_index`.
///
/// Each connected component is started from its lowest-degree vertex. Choosing
/// the start matters: beginning at a vertex in the middle of the graph gives a
/// much worse ordering than beginning at an extremity, and the lowest-degree
/// vertex is a cheap stand-in for an extremity.
pub fn reverse_cuthill_mckee(
  a : @csr.Csr[Double],
) -> Array[Int] raise @err.SparseError {
  if a.nrows() != a.ncols() {
    raise @err.SparseError::NotSquare(a.nrows(), a.ncols())
  }
  let n = a.nrows()
  if n == 0 {
    return []
  }
  let rp = a.row_ptr()
  let ci = a.col_idx()
  let deg = @graph.degree(a)
  let visited : Array[Bool] = Array::make(n, false)
  let order : Array[Int] = Array(capacity=n)
  // Queue reused across components; `head` walks it and `order` grows with it,
  // so the BFS order and the output order are the same array.
  let mut component = 0
  while component < n {
    // Start from the unvisited vertex of lowest degree.
    let mut start = -1
    let mut best_degree = 0
    for i = 0; i < n; i = i + 1 {
      if !visited[i] && (start < 0 || deg[i] < best_degree) {
        start = i
        best_degree = deg[i]
      }
    }
    if start < 0 {
      break
    }
    visited[start] = true
    order.push(start)
    let mut head = order.length() - 1
    while head < order.length() {
      let u = order[head]
      head = head + 1
      // Collect this vertex's unvisited neighbours, then take them in
      // increasing degree order.
      let neighbours : Array[Int] = Array(capacity=rp[u + 1] - rp[u])
      for k = rp[u]; k < rp[u + 1]; k = k + 1 {
        let v = ci[k]
        if v != u && !visited[v] {
          neighbours.push(v)
        }
      }
      neighbours.sort_by((x, y) => deg[x].compare(deg[y]))
      for v in neighbours {
        // A neighbour can be claimed by an earlier vertex in the same frontier,
        // so re-check rather than trusting the collection above.
        if !visited[v] {
          visited[v] = true
          order.push(v)
        }
      }
    }
    component = component + 1
  }
  // Reverse the breadth-first order — this is the R in RCM — and convert the
  // resulting sequence into a permutation vector. `order[pos]` is the vertex
  // that ends up at position `pos`, so inverting it gives `perm[old] = new`.
  let perm : Array[Int] = Array::make(n, 0)
  for pos = 0; pos < n; pos = pos + 1 {
    perm[order[n - 1 - pos]] = pos
  }
  perm
}

///|
/// Reorder a symmetric matrix by reverse Cuthill-McKee and return both the
/// reordered matrix and the permutation that produced it.
pub fn reorder_rcm(
  a : @csr.Csr[Double],
) -> (@csr.Csr[Double], Array[Int]) raise @err.SparseError {
  let perm = reverse_cuthill_mckee(a)
  (permute(a, perm), perm)
}
