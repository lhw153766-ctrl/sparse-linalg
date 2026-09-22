// The claim this package exists to make testable: the same `spgemm` computes
// reachability over one semiring and shortest paths over another, without any
// change to the storage or product code.

// Build an n x n matrix from weighted edges. Edges with no weight given are
// weight 1, which is what the Boolean semiring ignores anyway.

///|
fn from_edges(
  n : Int,
  edges : Array[(Int, Int, Double)],
) -> @csr.Csr[MinPlus] raise @err.SparseError {
  let coo : @coo.Coo[MinPlus] = @coo.Coo::empty(n, n)
  for e in edges {
    let (u, v, w) = e
    coo.push(u, v, MinPlus::new(w))
  }
  coo.to_csr()
}

///|
fn bool_from_edges(
  n : Int,
  edges : Array[(Int, Int)],
) -> @csr.Csr[BoolSemiring] raise @err.SparseError {
  let coo : @coo.Coo[BoolSemiring] = @coo.Coo::empty(n, n)
  for e in edges {
    let (u, v) = e
    coo.push(u, v, BoolSemiring::new(true))
  }
  coo.to_csr()
}

///|
test "the boolean semiring behaves as or and and" {
  let t = BoolSemiring::new(true)
  let f = BoolSemiring::new(false)
  inspect((t + f).value(), content="true")
  inspect((t * f).value(), content="false")
  inspect((t * t).value(), content="true")
  inspect((Default::default() : BoolSemiring).value(), content="false")
  // The additive identity really is an identity for this addition.
  inspect((f + t).value(), content="true")
  inspect((f + f).value(), content="false")
}

///|
test "the min-plus semiring behaves as minimum and addition" {
  inspect((MinPlus::new(3.0) + MinPlus::new(5.0)).value(), content="3")
  inspect((MinPlus::new(3.0) * MinPlus::new(5.0)).value(), content="8")
  // Default is infinity, which is the identity for minimum.
  inspect((Default::default() : MinPlus).is_infinite(), content="true")
  inspect((MinPlus::infinity() + MinPlus::new(7.0)).value(), content="7")
  // And one is the identity for addition.
  inspect((MinPlus::one() * MinPlus::new(7.0)).value(), content="7")
}

///|
test "spgemm over the boolean semiring computes two-step reachability" {
  // 0 -> 1 -> 2, and 3 is isolated.
  let a = bool_from_edges(4, [(0, 1), (1, 2)])
  let a2 = @ops.spgemm(a, a)
  // From 0 there is a walk of length two to 2, and none to 1 or 3.
  inspect(a2.get_or(0, 2, BoolSemiring::new(false)).value(), content="true")
  inspect(a2.get_or(0, 1, BoolSemiring::new(false)).value(), content="false")
  inspect(a2.get_or(0, 3, BoolSemiring::new(false)).value(), content="false")
  // 3 has no outgoing edges, so its row stays empty.
  inspect(a2.row_nnz(3), content="0")
}

///|
test "repeated squaring computes the transitive closure" {
  // A cycle 0 -> 1 -> 2 -> 0, plus a tail 2 -> 3, and 4 isolated.
  let a = bool_from_edges(5, [(0, 1), (1, 2), (2, 0), (2, 3)])
  // Square the accumulated relation, not the raw adjacency matrix.
  //
  // Accumulating the powers A, A^2, A^4, A^8 looks equivalent and is not: it
  // only covers path lengths that are powers of two, so the three-step walk
  // 0 -> 1 -> 2 -> 0 is missed. Squaring the accumulation covers every length
  // up to 2^k, which after three rounds is 8 steps and therefore every simple
  // path in a five-vertex graph.
  let mut reach = a
  for _round = 0; _round < 3; _round = _round + 1 {
    reach = @ops.add(reach, @ops.spgemm(reach, reach))
  }
  // Every vertex of the cycle reaches every other, and 3.
  for j = 0; j < 4; j = j + 1 {
    inspect(
      reach.get_or(0, j, BoolSemiring::new(false)).value(),
      content="true",
    )
    inspect(
      reach.get_or(1, j, BoolSemiring::new(false)).value(),
      content="true",
    )
  }
  // 3 has no outgoing edges and lies on no cycle, so its row stays empty —
  // including the diagonal. Reachability in this formulation means "there is a
  // walk of length at least one", not "at most one", so a vertex does not reach
  // itself unless it sits on a cycle.
  inspect(reach.row_nnz(3), content="0")
  inspect(reach.get_or(3, 3, BoolSemiring::new(false)).value(), content="false")
  inspect(reach.get_or(3, 0, BoolSemiring::new(false)).value(), content="false")
  // 4 is isolated for the same reason.
  inspect(reach.row_nnz(4), content="0")
  // Vertex 0 does lie on a cycle, so it does reach itself.
  inspect(reach.get_or(0, 0, BoolSemiring::new(false)).value(), content="true")
}

///|
test "spgemm over min-plus finds a shorter two-hop path" {
  // 0 -> 2 costs 5 directly, but 0 -> 1 -> 2 costs 1 + 2.
  let m = from_edges(3, [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)])
  let m2 = @ops.spgemm(m, m)
  inspect(m2.get_or(0, 2, MinPlus::infinity()).value(), content="3")
  // The direct edge is still there and still 5.
  inspect(m.get_or(0, 2, MinPlus::infinity()).value(), content="5")
}

///|
test "min-plus closure computes all-pairs shortest paths" {
  // A path 0 -1- 1 -4- 2 -1- 3, plus a shortcut 0 -> 3 of weight 20 that the
  // shortest route beats.
  let m = from_edges(4, [(0, 1, 1.0), (1, 2, 4.0), (2, 3, 1.0), (0, 3, 20.0)])
  // Same accumulation as above: square the running result so that paths of
  // every length are covered, not only lengths that are powers of two.
  let mut dist = m
  for _round = 0; _round < 3; _round = _round + 1 {
    dist = @ops.add(dist, @ops.spgemm(dist, dist))
  }
  inspect(dist.get_or(0, 3, MinPlus::infinity()).value(), content="6")
  inspect(dist.get_or(0, 2, MinPlus::infinity()).value(), content="5")
  inspect(dist.get_or(1, 3, MinPlus::infinity()).value(), content="5")
  // An unreachable vertex stays infinite rather than becoming zero, which is
  // the whole reason the additive identity has to be infinity.
  inspect(dist.get_or(3, 0, MinPlus::infinity()).is_infinite(), content="true")
}

///|
test "the same storage code carries both semirings" {
  // The point of the generic storage: nothing in Coo, Csr or the conversion
  // path knows or cares which of these it is holding.
  let b : @csr.Csr[BoolSemiring] = bool_from_edges(2, [(0, 1)])
  let m : @csr.Csr[MinPlus] = from_edges(2, [(0, 1, 2.5)])
  inspect(b.nnz(), content="1")
  inspect(m.nnz(), content="1")
  inspect(m.get_or(0, 1, MinPlus::infinity()).value(), content="2.5")
  // And the integer path keeps exactness, which no floating point type can.
  let i : @csr.Csr[Int64] = @csr.Csr::new(1, 2, [0, 2], [0, 1], [
    9000000000000L, 7L,
  ])
  inspect(i.get_or(0, 0, 0L) + i.get_or(0, 1, 0L), content="9000000000007")
}
