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
/// Right-pad a string to a fixed width so the columns line up.
fn pad(s : String, width : Int) -> String {
  let buf = StringBuilder::StringBuilder()
  buf.write_string(s)
  for _i = s.length(); _i < width; _i = _i + 1 {
    buf.write_char(' ')
  }
  buf.to_string()
}
