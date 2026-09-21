// Black-box tests for the graph constructions and algorithms.

///|
fn raises(build : () -> Unit raise @err.SparseError) -> Bool {
  try {
    build()
    false
  } catch {
    _ => true
  }
}

// A path 0-1-2-3, as an adjacency matrix.

///|
fn path4() -> @csr.Csr[Double] raise @err.SparseError {
  @graph.adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3)])
}

// Two disjoint triangles: vertices 0,1,2 and 3,4,5.

///|
fn two_triangles() -> @csr.Csr[Double] raise @err.SparseError {
  @graph.adjacency_from_edges(6, [
    (0, 1),
    (1, 2),
    (0, 2),
    (3, 4),
    (4, 5),
    (3, 5),
  ])
}

///|
test "degree counts stored neighbours and ignores the diagonal" {
  let a = path4()
  debug_inspect(@graph.degree(a), content="[1, 2, 2, 1]")
  // With a self-loop present the degree must not change.
  let with_loop : @csr.Csr[Double] = @csr.Csr::new(
    2,
    2,
    [0, 2, 4],
    [0, 1, 0, 1],
    [5.0, 1.0, 1.0, 5.0],
  )
  debug_inspect(@graph.degree(with_loop), content="[1, 1]")
}

///|
test "adjacency from an edge list is symmetric" {
  let a = path4()
  inspect(a.nnz(), content="6")
  inspect(a.is_symmetric(), content="true")
  debug_inspect(
    a.to_dense(),
    content="[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]",
  )
}

///|
test "a self-loop edge is stored once" {
  let a = @graph.adjacency_from_edges(2, [(0, 0), (0, 1)])
  // (0,0) once, plus (0,1) and (1,0).
  inspect(a.nnz(), content="3")
  debug_inspect(@graph.degree(a), content="[1, 1]")
}

///|
test "repeated edges accumulate weight" {
  let a = @graph.adjacency_from_edges(2, [(0, 1), (0, 1), (1, 0)])
  // The three pushes collapse to one coordinate with value 3.
  inspect(a.nnz(), content="2")
  inspect(a.get_or(0, 1, 0.0), content="3")
  inspect(a.get_or(1, 0, 0.0), content="3")
}

///|
test "weighted adjacency keeps the weights" {
  let a = @graph.weighted_adjacency_from_edges(3, [(0, 1, 2.5), (1, 2, -1.0)])
  inspect(a.get_or(0, 1, 0.0), content="2.5")
  inspect(a.get_or(1, 0, 0.0), content="2.5")
  inspect(a.get_or(1, 2, 0.0), content="-1")
}

///|
test "laplacian is D - A with zero row sums" {
  let a = path4()
  let l = @graph.laplacian(a)
  // Degrees along the diagonal: 1, 2, 2, 1.
  debug_inspect(l.diagonal(), content="[1, 2, 2, 1]")
  debug_inspect(
    l.to_dense(),
    content="[1, -1, 0, 0, -1, 2, -1, 0, 0, -1, 2, -1, 0, 0, -1, 1]",
  )
  inspect(l.is_symmetric(), content="true")
  // Every row must sum to zero: the constant vector is in the null space.
  let ones = [1.0, 1.0, 1.0, 1.0]
  let product = l.spmv(ones)
  for i = 0; i < 4; i = i + 1 {
    inspect(Double::abs(product[i]) < 1.0e-15, content="true")
  }
}

///|
test "laplacian of a weighted graph uses the weighted degree" {
  let a = @graph.weighted_adjacency_from_edges(2, [(0, 1, 3.0)])
  let l = @graph.laplacian(a)
  debug_inspect(l.diagonal(), content="[3, 3]")
  debug_inspect(l.to_dense(), content="[3, -3, -3, 3]")
}

///|
test "laplacian rejects a non-symmetric adjacency matrix" {
  // (0,1) is stored but (1,0) is not, so this is not symmetric.
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 1], [1], [1.0])
  inspect(
    raises(() => {
      let _ = @graph.laplacian(a)
    }),
    content="true",
  )
  let rect : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 1, 2], [0, 1], [
    1.0, 1.0,
  ])
  inspect(
    raises(() => {
      let _ = @graph.laplacian(rect)
    }),
    content="true",
  )
}

///|
test "normalized laplacian has unit diagonal on any graph" {
  let a = path4()
  let nl = @graph.normalized_laplacian(a)
  debug_inspect(nl.diagonal(), content="[1, 1, 1, 1]")
  inspect(nl.is_symmetric(), content="true")
  // On a path the endpoints have degree 1 and the interior vertices degree 2,
  // so the off-diagonal entries are -1/sqrt(1*2) and -1/sqrt(2*2).
  inspect(
    Double::abs(nl.get_or(0, 1, 0.0) + 1.0 / Double::sqrt(2.0)) < 1.0e-15,
    content="true",
  )
  inspect(Double::abs(nl.get_or(1, 2, 0.0) + 0.5) < 1.0e-15, content="true")
}

///|
test "normalized laplacian has zero row sums on a regular graph" {
  // Only for a regular graph does the normalized Laplacian annihilate the
  // constant vector; on a path the endpoints break the balance.
  let cycle = @graph.adjacency_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
  let nl = @graph.normalized_laplacian(cycle)
  debug_inspect(nl.diagonal(), content="[1, 1, 1, 1]")
  let ones = [1.0, 1.0, 1.0, 1.0]
  let product = nl.spmv(ones)
  for i = 0; i < 4; i = i + 1 {
    inspect(Double::abs(product[i]) < 1.0e-15, content="true")
  }
}

///|
test "normalized laplacian of a single edge" {
  let a = @graph.adjacency_from_edges(2, [(0, 1)])
  let nl = @graph.normalized_laplacian(a)
  // I - D^-1/2 A D^-1/2 with degree 1 on both ends gives [[1,-1],[-1,1]].
  debug_inspect(nl.to_dense(), content="[1, -1, -1, 1]")
}

///|
test "an isolated vertex has a zero-degree row" {
  // Vertex 2 has no edges.
  let a = @graph.adjacency_from_edges(3, [(0, 1)])
  debug_inspect(@graph.degree(a), content="[1, 1, 0]")
  let nl = @graph.normalized_laplacian(a)
  // The diagonal stays 1 rather than becoming a division by zero.
  debug_inspect(nl.diagonal(), content="[1, 1, 1]")
}

///|
test "connected components of a path" {
  let a = path4()
  debug_inspect(@graph.connected_components(a), content="[0, 0, 0, 0]")
  inspect(@graph.component_count(a), content="1")
  inspect(@graph.is_connected(a), content="true")
}

///|
test "connected components of two disjoint triangles" {
  let a = two_triangles()
  debug_inspect(@graph.connected_components(a), content="[0, 0, 0, 1, 1, 1]")
  inspect(@graph.component_count(a), content="2")
  inspect(@graph.is_connected(a), content="false")
}

///|
test "an edgeless graph has one component per vertex" {
  let a : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 0, 0, 0], [], [])
  inspect(@graph.component_count(a), content="3")
  inspect(@graph.is_connected(a), content="false")
}

///|
test "the empty graph is not connected" {
  let a : @csr.Csr[Double] = @csr.Csr::new(0, 0, [0], [], [])
  inspect(@graph.is_connected(a), content="false")
  debug_inspect(@graph.connected_components(a), content="[]")
}

///|
test "page rank on a symmetric cycle is uniform" {
  // 0 -> 1 -> 2 -> 0: every vertex is equivalent, so each holds a third.
  let a : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 1, 2, 3], [1, 2, 0], [
    1.0, 1.0, 1.0,
  ])
  let r = @graph.page_rank(a)
  for i = 0; i < 3; i = i + 1 {
    inspect(Double::abs(r[i] - 1.0 / 3.0) < 1.0e-9, content="true")
  }
}

///|
test "page rank sums to one and handles dangling vertices" {
  // 0 links to 1 and 2; 1 and 2 have no outgoing links, so their rank would
  // leak without the dangling redistribution.
  let a : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 2, 2, 2], [1, 2], [
    1.0, 1.0,
  ])
  let r = @graph.page_rank(a)
  let mut total = 0.0
  for v in r {
    total = total + v
  }
  inspect(Double::abs(total - 1.0) < 1.0e-12, content="true")
  // The fixed point is r0 = (0.15/3 + 0.85/3) / (1 + 0.85/3), and the two
  // dangling vertices share the rest equally.
  inspect(Double::abs(r[0] - 0.2597402597402597) < 1.0e-9, content="true")
  inspect(Double::abs(r[1] - 0.37012987012987015) < 1.0e-9, content="true")
  inspect(Double::abs(r[2] - 0.37012987012987015) < 1.0e-9, content="true")
}

///|
test "page rank on a star puts the leaves above the hub" {
  let a : @csr.Csr[Double] = @csr.Csr::new(3, 3, [0, 2, 2, 2], [1, 2], [
    1.0, 1.0,
  ])
  let r = @graph.page_rank(a)
  inspect(r[0] < r[1], content="true")
}

///|
test "page rank rejects invalid damping and non-square input" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 2], [1, 0], [1.0, 1.0])
  inspect(
    raises(() => {
      let _ = @graph.page_rank(a, damping=0.0)
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = @graph.page_rank(a, damping=1.0)
    }),
    content="true",
  )
  inspect(
    raises(() => {
      let _ = @graph.page_rank(a, damping=-0.5)
    }),
    content="true",
  )
  let rect : @csr.Csr[Double] = @csr.Csr::new(2, 3, [0, 1, 2], [0, 1], [
    1.0, 1.0,
  ])
  inspect(
    raises(() => {
      let _ = @graph.page_rank(rect)
    }),
    content="true",
  )
}

///|
test "page rank of the empty graph" {
  let a : @csr.Csr[Double] = @csr.Csr::new(0, 0, [0], [], [])
  debug_inspect(@graph.page_rank(a), content="[]")
}

///|
test "jacobi sweeps converge on a diagonally dominant system" {
  // 3x3 with a strong diagonal.
  let a : @csr.Csr[Double] = @csr.Csr::new(
    3,
    3,
    [0, 2, 4, 6],
    [0, 1, 0, 1, 1, 2],
    [4.0, -1.0, -1.0, 4.0, -1.0, 4.0],
  )
  let b = [1.0, 2.0, 3.0]
  let x = @graph.jacobi_sweeps(a, b, sweeps=200)
  inspect(@solve.relative_residual(a, x, b) < 1.0e-6, content="true")
}

///|
test "jacobi sweeps reject a zero diagonal" {
  let a : @csr.Csr[Double] = @csr.Csr::new(2, 2, [0, 1, 1], [0], [0.0])
  inspect(
    raises(() => {
      let _ = @graph.jacobi_sweeps(a, [1.0, 1.0])
    }),
    content="true",
  )
}

///|
test "the laplacian is what cg and cholesky expect" {
  // Pinning the first vertex (a Dirichlet boundary condition) makes the
  // Laplacian positive definite, which is the usual way to make a Laplacian
  // system solvable.
  let a = path4()
  let l = @graph.laplacian(a)
  let n = 4
  let coo : @coo.Coo[Double] = @coo.Coo::empty(n, n)
  for i = 1; i < n; i = i + 1 {
    for j = 1; j < n; j = j + 1 {
      let v = l.get_or(i, j, 0.0)
      if v != 0.0 {
        coo.push(i, j, v)
      }
    }
  }
  coo.push(0, 0, 1.0)
  let pinned = coo.to_csr()
  let b = [1.0, 1.0, 1.0, 1.0]
  let direct = @solve.solve_cholesky(pinned, b)
  let iterative = @solve.cg(pinned, b)
  inspect(iterative.converged, content="true")
  for i = 0; i < n; i = i + 1 {
    inspect(Double::abs(direct[i] - iterative.x[i]) < 1.0e-9, content="true")
  }
}
