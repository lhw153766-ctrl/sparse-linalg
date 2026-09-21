// Black-box tests for the reverse Cuthill-McKee ordering.
//
// The interesting claim is not that the permutation is valid — that is easy —
// but that it actually reduces the bandwidth and, with it, the fill-in of a
// Cholesky factorisation. Both are measured before and after.

///|
fn raises(build : () -> Unit raise @err.SparseError) -> Bool {
  try {
    build()
    false
  } catch {
    _ => true
  }
}

// A k by k grid graph: k*k vertices, each joined to its right and lower
// neighbour. Under the natural row-major numbering its Laplacian has bandwidth
// k, which is what makes it a useful test case.

///|
fn grid_graph(k : Int) -> @csr.Csr[Double] raise @err.SparseError {
  let n = k * k
  let edges : Array[(Int, Int)] = Array(capacity=2 * k * (k - 1))
  for r = 0; r < k; r = r + 1 {
    for c = 0; c < k; c = c + 1 {
      let v = r * k + c
      if c + 1 < k {
        edges.push((v, v + 1))
      }
      if r + 1 < k {
        edges.push((v, v + k))
      }
    }
  }
  @graph.adjacency_from_edges(n, edges)
}

// A deterministic Fisher-Yates shuffle. Deterministic on purpose: a failing
// test has to be reproducible. The multiplier and modulus are kept small enough
// that the arithmetic cannot overflow a 32-bit Int.

///|
fn scramble(n : Int) -> Array[Int] {
  let perm : Array[Int] = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    perm[i] = i
  }
  let mut seed = 12345
  for i = n - 1; i > 0; i = i - 1 {
    seed = (seed * 1103 + 12345) % 1000003
    let j = seed % (i + 1)
    let tmp = perm[i]
    perm[i] = perm[j]
    perm[j] = tmp
  }
  perm
}

///|
test "bandwidth and profile of a banded matrix" {
  let l = @graph.laplacian(grid_graph(4))
  // Row 0 is adjacent to 1 and 4, so the widest offset is 4.
  inspect(@reorder.bandwidth(l), content="4")
  inspect(@reorder.profile(l) > 0, content="true")
}

///|
test "bandwidth of a diagonal matrix is zero" {
  let a : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 1, 2, 3], [0, 1, 2], [
    1.0, 1.0, 1.0,
  ])
  inspect(@reorder.bandwidth(a), content="0")
  inspect(@reorder.profile(a), content="0")
}

///|
test "is_permutation and invert_permutation" {
  inspect(@reorder.is_permutation([2, 0, 1], 3), content="true")
  inspect(@reorder.is_permutation([2, 0, 2], 3), content="false")
  inspect(@reorder.is_permutation([2, 0], 3), content="false")
  inspect(@reorder.is_permutation([0, 1, 3], 3), content="false")
  inspect(@reorder.is_permutation([0, 1, -1], 3), content="false")
  debug_inspect(@reorder.invert_permutation([2, 0, 1]), content="[1, 2, 0]")
  inspect(
    raises(() => {
      let _ = @reorder.invert_permutation([0, 0])
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = @reorder.invert_permutation([0, 5])
    }),
    content="true",
  )
}

///|
test "inverting twice recovers the permutation" {
  let perm = [3, 0, 4, 1, 2]
  let back = @reorder.invert_permutation(@reorder.invert_permutation(perm))
  debug_inspect(back, content="[3, 0, 4, 1, 2]")
}

///|
test "permute preserves the pattern size, trace and Frobenius norm" {
  let l = @graph.laplacian(grid_graph(3))
  let perm = scramble(9)
  let p = @reorder.permute(l, perm)
  inspect(p.nnz(), content=l.nnz().to_string())
  inspect(p.is_symmetric(), content="true")
  // A relabelling changes nothing about the matrix but the order of the
  // unknowns, so these invariants must hold exactly.
  let mut trace_before = 0.0
  let mut trace_after = 0.0
  let d1 = l.diagonal()
  let d2 = p.diagonal()
  for i = 0; i < 9; i = i + 1 {
    trace_before = trace_before + d1[i]
    trace_after = trace_after + d2[i]
  }
  inspect(Double::abs(trace_before - trace_after) < 1.0e-12, content="true")
  let mut norm_before = 0.0
  let mut norm_after = 0.0
  for v in l.vals() {
    norm_before = norm_before + v * v
  }
  for v in p.vals() {
    norm_after = norm_after + v * v
  }
  inspect(Double::abs(norm_before - norm_after) < 1.0e-9, content="true")
}

///|
test "permute by the identity changes nothing" {
  let l = @graph.laplacian(grid_graph(3))
  let same = @reorder.permute(l, [0, 1, 2, 3, 4, 5, 6, 7, 8])
  inspect(same.nnz(), content=l.nnz().to_string())
  for i = 0; i < 9; i = i + 1 {
    for j = 0; j < 9; j = j + 1 {
      inspect(same.get_or(i, j, 0.0), content=l.get_or(i, j, 0.0).to_string())
    }
  }
}

///|
test "permute rejects something that is not a permutation" {
  let l = @graph.laplacian(grid_graph(2))
  inspect(
    raises(() => {
      let _ = @reorder.permute(l, [0, 0, 1, 2])
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = @reorder.permute(l, [0, 1, 2])
    }),
    content="true",
  )
}

///|
test "rcm returns a permutation" {
  let l = @graph.laplacian(grid_graph(5))
  let perm = @reorder.reverse_cuthill_mckee(l)
  inspect(@reorder.is_permutation(perm, 25), content="true")
}

///|
test "rcm reduces the bandwidth of a scrambled grid" {
  let l = @graph.laplacian(grid_graph(5))
  let scrambled = @reorder.permute(l, scramble(25))
  let before = @reorder.bandwidth(scrambled)
  let (ordered, _perm) = @reorder.reorder_rcm(scrambled)
  let after = @reorder.bandwidth(ordered)
  // Scrambling spreads the entries over the whole matrix; RCM has to bring
  // them back towards the diagonal.
  inspect(before > 5, content="true")
  inspect(after < before, content="true")
}

///|
test "rcm reduces the profile of a scrambled grid" {
  let l = @graph.laplacian(grid_graph(5))
  let scrambled = @reorder.permute(l, scramble(25))
  let (ordered, _perm) = @reorder.reorder_rcm(scrambled)
  inspect(
    @reorder.profile(ordered) < @reorder.profile(scrambled),
    content="true",
  )
}

///|
test "rcm on a disconnected graph still returns every vertex once" {
  // Two triangles, numbered so the components interleave.
  let a = @graph.adjacency_from_edges(6, [
    (0, 3),
    (3, 4),
    (0, 4),
    (1, 2),
    (2, 5),
    (1, 5),
  ])
  let perm = @reorder.reverse_cuthill_mckee(a)
  inspect(@reorder.is_permutation(perm, 6), content="true")
}

///|
test "rcm on a graph with an isolated vertex" {
  let a = @graph.adjacency_from_edges(3, [(0, 1)])
  let perm = @reorder.reverse_cuthill_mckee(a)
  inspect(@reorder.is_permutation(perm, 3), content="true")
}

///|
test "rcm on an empty and a diagonal matrix" {
  let empty : @csr.Csr[Double] = @csr.Csr::new(0, 0, [0], [], [])
  debug_inspect(@reorder.reverse_cuthill_mckee(empty), content="[]")
  let diag : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 1, 2, 3], [0, 1, 2], [
    1.0, 2.0, 3.0,
  ])
  inspect(
    @reorder.is_permutation(@reorder.reverse_cuthill_mckee(diag), 3),
    content="true",
  )
  // A diagonal matrix is already as narrow as it can be.
  inspect(@reorder.bandwidth(diag), content="0")
}

///|
test "rcm rejects a non-square matrix" {
  let rect : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 1, 2], [0, 1], [
    1.0, 1.0,
  ])
  inspect(
    raises(() => {
      let _ = @reorder.reverse_cuthill_mckee(rect)
    }),
    content="true",
  )
}
