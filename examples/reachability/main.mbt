// The same matrix product, two different algorithms.
//
// A sparse matrix product only asks its scalar type for two operations: one
// that plays the role of addition, and one that plays the role of
// multiplication. Nothing in the storage layout or in the index arithmetic
// cares what those operations are.
//
// Change them and the same `spgemm` computes something else:
//
//   Boolean semiring   add = OR,  multiply = AND   ->  reachability
//   min-plus semiring  add = min, multiply = plus  ->  shortest paths
//
// This example runs both over the same storage code. Neither the `Coo`
// assembly, the `Csr` storage, nor `spgemm` is specialised for graphs, and none
// of them knows which semiring it is holding.
//
// Run from the project root:
//
//   moon run examples/reachability

///|
/// Vertices of the small directed graph used by both halves of the example.
fn names() -> Array[String] {
  ["a", "b", "c", "d", "e", "f"]
}

///|
fn main raise @err.SparseError {
  reachability()
  println("")
  shortest_paths()
}

///|
/// Transitive closure over the Boolean semiring.
fn reachability() -> Unit raise @err.SparseError {
  // a to b to c to a is a cycle; c to d hangs off it; e to f is separate.
  let edges : Array[(Int, Int)] = [(0, 1), (1, 2), (2, 0), (2, 3), (4, 5)]
  let adjacency = bool_adjacency(edges)
  println("reachability, computed by squaring the adjacency matrix")
  println("  edges: " + edge_list(edges))
  println("")
  // Squaring the accumulated relation covers every walk length up to 2^k. It is
  // not enough to accumulate the powers A, A^2, A^4: that would miss the
  // three-step walk a to b to c to a, whose length is not a power of two.
  let mut reach = adjacency
  let mut rounds = 0
  while rounds < 4 {
    reach = @ops.add(reach, @ops.spgemm(reach, reach))
    rounds = rounds + 1
  }
  print_matrix("  row reaches column", reach, v => {
    if v.value() {
      "1"
    } else {
      "0"
    }
  })
  println("")
  println(
    "  a, b and c are on a cycle, so each reaches all three of them plus d.",
  )
  println("  d has no outgoing edge, so its row stays empty.")
  println("  e reaches f; f reaches nothing, not even itself, because a vertex")
  println("  only reaches itself when it sits on a cycle.")
}

///|
/// All-pairs shortest paths over the min-plus semiring.
fn shortest_paths() -> Unit raise @err.SparseError {
  // A weighted directed graph. The direct a to d edge is expensive; going via
  // b and c is cheaper, and only a closure finds that out.
  let edges : Array[(Int, Int, Double)] = [
    (0, 1, 1.0),
    (1, 2, 2.0),
    (2, 3, 1.0),
    (0, 3, 9.0),
    (1, 4, 7.0),
    (4, 5, 1.0),
  ]
  let weights = weighted_adjacency(edges)
  println("shortest paths, computed by the same product over min-plus")
  println("  weighted edges: " + weighted_edge_list(edges))
  println("")
  println(
    "  the direct a to d edge costs 9, but a to b to c to d costs 1+2+1 = 4",
  )
  println("")
  let mut dist = weights
  let mut rounds = 0
  while rounds < 4 {
    dist = @ops.add(dist, @ops.spgemm(dist, dist))
    rounds = rounds + 1
  }
  print_matrix("  row to column", dist, v => {
    if v.is_infinite() {
      "inf"
    } else {
      v.value().to_string()
    }
  })
  println("")
  println("  unreachable pairs print as inf, the additive identity of this")
  println(
    "  semiring. Since min(inf, x) is x, an absent edge means \"no route\"",
  )
  println("  rather than \"zero cost\", which is what makes the same product")
  println("  compute shortest paths instead of matrix multiplication.")
}

///|
/// Build an unweighted adjacency matrix over the Boolean semiring.
fn bool_adjacency(
  edges : Array[(Int, Int)],
) -> @csr.Csr[@semiring.BoolSemiring] raise @err.SparseError {
  let n = names().length()
  let coo : @coo.Coo[@semiring.BoolSemiring] = @coo.Coo::empty(n, n)
  for e in edges {
    let (u, v) = e
    coo.push(u, v, @semiring.BoolSemiring::new(true))
  }
  coo.to_csr()
}

///|
/// Build a weighted adjacency matrix over the min-plus semiring.
fn weighted_adjacency(
  edges : Array[(Int, Int, Double)],
) -> @csr.Csr[@semiring.MinPlus] raise @err.SparseError {
  let n = names().length()
  let coo : @coo.Coo[@semiring.MinPlus] = @coo.Coo::empty(n, n)
  for e in edges {
    let (u, v, w) = e
    coo.push(u, v, @semiring.MinPlus::new(w))
  }
  coo.to_csr()
}

///|
fn edge_list(edges : Array[(Int, Int)]) -> String {
  let buf = StringBuilder::StringBuilder()
  for e in edges {
    let (u, v) = e
    buf.write_string(names()[u] + "->" + names()[v] + " ")
  }
  buf.to_string()
}

///|
fn weighted_edge_list(edges : Array[(Int, Int, Double)]) -> String {
  let buf = StringBuilder::StringBuilder()
  for e in edges {
    let (u, v, w) = e
    buf.write_string(
      names()[u] + "->" + names()[v] + "(" + w.to_string() + ") ",
    )
  }
  buf.to_string()
}

// Render a sparse matrix densely, with a formatter for the values so that both
// semirings share the routine. The fact that this works for either type is the
// same point the example is making.

///|
fn[F] print_matrix(
  label : String,
  a : @csr.Csr[F],
  show : (F) -> String,
) -> Unit raise @err.SparseError {
  let header = StringBuilder::StringBuilder()
  header.write_string("        ")
  for j = 0; j < names().length(); j = j + 1 {
    header.write_string("  " + names()[j] + " ")
  }
  println(label)
  println(header.to_string())
  for i = 0; i < a.nrows(); i = i + 1 {
    let row = StringBuilder::StringBuilder()
    row.write_string("    " + names()[i] + " ")
    for j = 0; j < a.ncols(); j = j + 1 {
      match a.get(i, j) {
        Some(v) => row.write_string("  " + show(v) + " ")
        None => row.write_string("  .  ")
      }
    }
    println(row.to_string())
  }
}
