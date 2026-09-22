// Benchmarks for the sparse products.
//
// `spgemm` is the one worth measuring: it is the operation that turns a graph's
// adjacency matrix into reachability or shortest paths, and its cost is driven
// by how much the product fills in rather than by the size of the operands.
//
// Run with:
//
//   moon bench

// The 5-point stencil on a k x k grid: a sparse matrix whose square is denser
// than itself.
///|
fn bench_grid(k : Int) -> @csr.Csr[Double] {
  let n = k * k
  let rp : Array[Int] = Array::make(n + 1, 0)
  let ci : Array[Int] = Array(capacity=5 * n)
  let va : Array[Double] = Array(capacity=5 * n)
  for r = 0; r < k; r = r + 1 {
    for c = 0; c < k; c = c + 1 {
      let v = r * k + c
      let cols : Array[Int] = [v]
      if r > 0 {
        cols.push(v - k)
      }
      if r + 1 < k {
        cols.push(v + k)
      }
      if c > 0 {
        cols.push(v - 1)
      }
      if c + 1 < k {
        cols.push(v + 1)
      }
      cols.sort()
      for col in cols {
        ci.push(col)
        va.push(if col == v { 4.0 } else { -1.0 })
      }
      rp[v + 1] = ci.length()
    }
  }
  @csr.Csr::new_unchecked(n, n, rp, ci, va)
}

///|
test "bench products" (b : @bench.T) {
  let k = 30
  let a = bench_grid(k)
  let n = k * k
  let x : Array[Double] = Array::make(n, 1.0)
  let dense : Array[Double] = Array::make(n * n, 0.0)
  b.bench(name="spmv, 900 unknowns", fn() {
    b.keep(a.spmv(x)) catch {
      _ => ()
    }
  })
  b.bench(name="spmm against a 1-column dense block", fn() {
    b.keep(@ops.spmm(a, dense, n)) catch {
      _ => ()
    }
  })
  b.bench(name="spgemm, 900x900 grid stencil squared", fn() {
    b.keep(@ops.spgemm(a, a).nnz()) catch {
      _ => ()
    }
  })
  b.bench(name="sparse add of two grid stencils", fn() {
    b.keep(@ops.add(a, a).nnz()) catch {
      _ => ()
    }
  })
}

///|
test "bench spgemm fill-in" (b : @bench.T) {
  // The product of a graph adjacency matrix with itself is denser than the
  // input, and how much denser is the interesting number. It is reported by the
  // test rather than asserted, because it depends on the graph.
  let k = 30
  let a = bench_grid(k)
  let squared = @ops.spgemm(a, a)
  println("grid stencil: nnz \{a.nnz()} squared to \{squared.nnz()}")
  b.bench(name="spgemm again, warmed allocator", fn() {
    b.keep(@ops.spgemm(a, a).nnz()) catch {
      _ => ()
    }
  })
}
