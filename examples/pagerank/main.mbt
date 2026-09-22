// PageRank on a small web graph, by sparse power iteration.
//
// PageRank is the classic example of a problem that looks like it needs linear
// algebra but reduces to one sparse product per iteration:
//
//   r <- (1 - d) / n + d * A^T (r / outdegree)
//
// where `A` is the link matrix. The dense form of a web graph is hopeless —
// a million pages would need a million-squared matrix — while the sparse form
// holds only the links that actually exist.
//
// The graph below is small enough that the answer can be checked by reasoning
// about it, which is the point: the ranks are printed alongside the in-degree
// and out-degree of every page so the result can be argued with.
//
// Run from the project root:
//
//   moon run examples/pagerank

///|
fn main raise @err.SparseError {
  small_graph()
  println("")
  large_graph()
}

///|
/// A graph small enough to check by hand.
fn small_graph() -> Unit raise @err.SparseError {
  // Pages, and the links between them. Index 0 is the home page.
  let names = [
    "home", "docs", "blog", "about", "pricing", "signup", "contact", "orphan",
  ]
  // (from, to) pairs.
  let links = [
    (0, 1),
    (0, 2),
    (0, 4),
    (0, 5),
    (1, 0),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 3),
    (3, 0),
    (3, 6),
    (4, 0),
    (4, 5),
    (5, 3),
    (6, 0),
    (6, 3),
  ]
  let n = names.length()
  let coo : @coo.Coo[Double] = @coo.Coo::empty(n, n)
  for l in links {
    let (from, to) = l
    coo.push(from, to, 1.0)
  }
  let a = coo.to_csr()

  println("PageRank over \{n} pages and \{links.length()} links")
  println("")
  println("page       out  in   rank")
  let ranks = @graph.page_rank(a)
  let in_degree : Array[Int] = Array::make(n, 0)
  for l in links {
    let (_, to) = l
    in_degree[to] = in_degree[to] + 1
  }
  let out_degree : Array[Int] = Array::make(n, 0)
  for l in links {
    let (from, _) = l
    out_degree[from] = out_degree[from] + 1
  }
  for i = 0; i < n; i = i + 1 {
    let name = names[i]
    println(
      "\{pad(name, 10)} \{out_degree[i]}    \{in_degree[i]}    \{ranks[i]}",
    )
  }
  println("")
  let mut total = 0.0
  for r in ranks {
    total = total + r
  }
  println(
    "the ranks sum to \{total}, which is what a probability distribution should do",
  )
  println("")
  println(
    "\"orphan\" has no incoming links and is not linked to by anything, so it",
  )
  println(
    "keeps only the damping floor; the pages reachable from \"home\" rank higher.",
  )
}

///|
/// A few hundred pages, generated deterministically.
///
/// The point of this half is scale rather than legibility: PageRank is a power
/// iteration whose every step is one sparse product, so a graph with a few
/// hundred vertices and a few thousand links is no harder than the eight-page
/// one above. A dense representation of this graph would hold 250,000 entries
/// against the 2,000-odd that are actually there.
fn large_graph() -> Unit raise @err.SparseError {
  let n = 500
  let links : Array[(Int, Int)] = Array(capacity=2500)
  // A deterministic pseudo-random graph: each page links to a handful of
  // others, biased towards low-numbered pages so that a clear ranking emerges
  // instead of a flat distribution.
  let mut seed = 987654321
  for from = 0; from < n; from = from + 1 {
    let out_degree = 3 + from % 5
    for _k = 0; _k < out_degree; _k = _k + 1 {
      seed = (seed * 1103515245 + 12345) % 2147483647
      // Square the uniform draw to bias towards early pages.
      let u = seed.to_double() / 2147483647.0
      let to = (u * u * n.to_double()).to_int()
      if to != from {
        links.push((from, to))
      }
    }
  }
  let coo : @coo.Coo[Double] = @coo.Coo::empty(n, n)
  for l in links {
    let (from, to) = l
    coo.push(from, to, 1.0)
  }
  let a = coo.to_csr()
  let dense_entries = n * n
  println("a generated graph of \{n} pages and \{links.length()} links")
  println(
    "  stored entries      : \{a.nnz()}  (dense would hold \{dense_entries})",
  )
  let ranks = @graph.page_rank(a)
  let mut total = 0.0
  for r in ranks {
    total = total + r
  }
  println("  ranks sum to        : \{total}")
  println("")
  println("  the ten highest-ranked pages:")
  // Rank the pages by score without a sorting library: repeatedly take the best
  // remaining one, which is fine for a demonstration of ten.
  let taken : Array[Bool] = Array::make(n, false)
  for _slot = 0; _slot < 10; _slot = _slot + 1 {
    let mut best = -1
    for i = 0; i < n; i = i + 1 {
      if !taken[i] && (best < 0 || ranks[i] > ranks[best]) {
        best = i
      }
    }
    if best < 0 {
      break
    }
    taken[best] = true
    println("    page \{pad(best.to_string(), 4)}  rank \{ranks[best]}")
  }
}

///|
/// Right-pad a string to a fixed width so the columns line up.
fn pad(s : String, width : Int) -> String {
  let buf = StringBuilder::StringBuilder()
  buf.write_string(s)
  for _i = s.length(); _i < width; _i = _i + 1 {
    buf.write_char(' ')
  }
  buf.to_string()
}
