// Benchmarks for the iterative solvers, and for what a warm start buys.
//
// The warm start is the interesting one. A time-stepping loop solves a sequence
// of systems whose solution barely changes between steps, so starting from the
// previous answer should cut the iteration count. These benchmarks measure
// whether it does, on a grid where the answer is not obvious in advance.
//
// Run with:
//
//   moon bench
//
// `bench` takes a non-raising closure, so the bodies wrap their calls. The
// dimension checks being wrapped are O(1).

// The 5-point stencil on a k x k grid, as CSR arrays.
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
test "bench cold and warm" (b : @bench.T) {
  let k = 40
  let a = bench_grid(k)
  let n = k * k
  let rhs : Array[Double] = Array::make(n, 1.0)
  let opts = @solve.SolverOptions::new(1.0e-10, 5000)
  // Solve once to get an answer to warm start from.
  let cold = @solve.cg(a, rhs, options=opts)
  let previous : Array[Double] = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    previous[i] = cold.x[i] * 1.000001
  }
  b.bench(name="cg cold start, 1600 unknowns", fn() {
    b.keep(@solve.cg(a, rhs, options=opts).iterations) catch {
      _ => ()
    }
  })
  b.bench(name="cg warm start, same system", fn() {
    b.keep(@solve.cg(a, rhs, options=opts, initial=Some(previous)).iterations) catch {
      _ => ()
    }
  })
  b.bench(name="cg cold, ilu0 preconditioned", fn() {
    try {
      let p = @solve.Precond::Ilu0(@solve.Ilu0Precond::new(a))
      b.keep(@solve.cg(a, rhs, options=opts, precond=p).iterations)
    } catch {
      _ => ()
    }
  })
}

///|
test "bench factorisation" (b : @bench.T) {
  let k = 30
  let a = bench_grid(k)
  let n = k * k
  let rhs : Array[Double] = Array::make(n, 1.0)
  b.bench(name="sparse cholesky factorise, 900 unknowns", fn() {
    b.keep(@solve.CholeskyFactor::new(a).nnz()) catch {
      _ => ()
    }
  })
  // The factorisation is hoisted out of the timed closure: measuring "solve
  // after factorising" while factorising inside the loop would just measure the
  // factorisation again.
  let factor = @solve.CholeskyFactor::new(a)
  b.bench(name="cholesky solve, factor already computed", fn() {
    b.keep(factor.solve(rhs)) catch {
      _ => ()
    }
  })
}
