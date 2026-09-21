// Benchmarks for the sparse kernels.
//
// The point is not to produce a headline number but to show that the cost is
// proportional to the number of stored entries rather than to the size of the
// dense matrix the same operator would have had. The 10000-point case has ten
// times the stored entries of the 1000-point case and should take roughly ten
// times as long, even though its dense counterpart would be a hundred times
// bigger.
//
// Run with:
//
//   moon bench
//
// `bench` takes a non-raising closure, so each body wraps its call. The
// dimension check being wrapped is `O(1)` and does not affect what is measured.

///|
fn stencil_matrix(n : Int) -> @csr.Csr[Double] {
  let rp : Array[Int] = Array::make(n + 1, 0)
  let ci : Array[Int] = Array::make(3 * n - 2, 0)
  let va : Array[Double] = Array::make(3 * n - 2, 0.0)
  let mut k = 0
  for i = 0; i < n; i = i + 1 {
    if i > 0 {
      ci[k] = i - 1
      va[k] = -1.0
      k = k + 1
    }
    ci[k] = i
    va[k] = 2.0
    k = k + 1
    if i + 1 < n {
      ci[k] = i + 1
      va[k] = -1.0
      k = k + 1
    }
    rp[i + 1] = k
  }
  @csr.Csr::new_unchecked(n, n, rp, ci, va)
}

///|
test "bench 1000" (b : @bench.T) {
  // 1000x1000, tridiagonal: 2998 stored entries against 1,000,000 dense ones.
  let n = 1000
  let a = stencil_matrix(n)
  let x : Array[Double] = Array::make(n, 1.0)
  let out : Array[Double] = Array::make(n, 0.0)
  let csc = @csc.Csc::from_csr(a)
  b.bench(name="spmv csr 1000x1000", fn() {
    b.keep(a.spmv(x)) catch {
      _ => ()
    }
  })
  b.bench(name="spmv csc 1000x1000", fn() {
    b.keep(csc.spmv(x)) catch {
      _ => ()
    }
  })
  b.bench(name="spmv_transpose csc 1000x1000", fn() {
    b.keep(csc.spmv_transpose(x)) catch {
      _ => ()
    }
  })
  b.bench(name="spmv_add 1000x1000", fn() {
    a.spmv_add(1.0, x, 0.0, out) catch {
      _ => ()
    }
  })
  b.bench(name="binary search lookup 1000 entries", fn() {
    try {
      let mut acc = 0.0
      for i = 0; i < n; i = i + 1 {
        acc = acc + a.get_or(i, i * 7 % n, 0.0)
      }
      b.keep(acc)
    } catch {
      _ => ()
    }
  })
}

///|
test "bench 10000" (b : @bench.T) {
  // Ten times the stored entries. The dense counterpart of this shape would
  // hold 100,000,000 entries.
  let n = 10000
  let a = stencil_matrix(n)
  let x : Array[Double] = Array::make(n, 1.0)
  b.bench(name="spmv csr 10000x10000", fn() {
    b.keep(a.spmv(x)) catch {
      _ => ()
    }
  })
}
