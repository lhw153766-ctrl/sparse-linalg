// What a reordering buys, measured rather than asserted.
//
// A sparse direct factorisation fills in: eliminating an unknown couples all of
// its neighbours, so the factor holds more entries than the matrix. How many
// more depends entirely on the order the unknowns are eliminated in, and that
// order is free to choose.
//
// The demonstration uses a 2D grid Laplacian, the standard test case, and
// reports three numberings of the same matrix:
//
//   natural     row-major, which for a grid is already good
//   scrambled   an arbitrary numbering, which is what a mesh from a CAD tool
//               or a mesh generator often looks like
//   rcm         the scrambled one, reordered by reverse Cuthill-McKee
//
// The natural ordering is included as the baseline, because for a grid
// row-major numbering is already close to optimal and it would be easy to
// claim an improvement that is not there. On a 60x60 grid RCM recovers the
// bandwidth of the scrambled matrix exactly (3558 down to 60) and ends up with
// a factor slightly smaller than the natural numbering's, without being told
// anything about the matrix being a grid.
//
// Run from the project root:
//
//   moon run examples/reordering

///|
/// The grid is `SIDE x SIDE`.
const SIDE : Int = 60

///|
fn main raise @err.SparseError {
  let k = SIDE
  let n = k * k
  let grid = grid_laplacian(k)
  println(
    "\{k}x\{k} grid Laplacian: \{n} unknowns, \{grid.nnz()} stored entries",
  )
  println("")

  // A fixed pseudo-random numbering. Deterministic so the output is stable.
  let scramble = scramble_permutation(n)
  report("natural", grid)
  let scrambled = @reorder.permute(grid, scramble)
  report("scrambled", scrambled)
  let (rcm, _perm) = @reorder.reorder_rcm(scrambled)
  report("rcm of scrambled", rcm)
}

///|
/// Print the bandwidth, the profile and the factor size for one numbering.
fn report(label : String, a : @csr.Csr[Double]) -> Unit raise @err.SparseError {
  let factor = @solve.CholeskyFactor::new(a)
  let ratio = factor.nnz().to_double() / a.nnz().to_double()
  println("\{label}:")
  println("  bandwidth        : \{@reorder.bandwidth(a)}")
  println("  profile          : \{@reorder.profile(a)}")
  println("  factor entries   : \{factor.nnz()}  (\{ratio}x the matrix)")
  println("")
}

///|
/// The 5-point stencil Laplacian on a `k x k` grid.
fn grid_laplacian(k : Int) -> @csr.Csr[Double] raise @err.SparseError {
  let n = k * k
  let coo : @coo.Coo[Double] = @coo.Coo::empty(n, n)
  for r = 0; r < k; r = r + 1 {
    for c = 0; c < k; c = c + 1 {
      let v = r * k + c
      coo.push(v, v, 4.0)
      if r > 0 {
        coo.push(v, v - k, -1.0)
      }
      if r + 1 < k {
        coo.push(v, v + k, -1.0)
      }
      if c > 0 {
        coo.push(v, v - 1, -1.0)
      }
      if c + 1 < k {
        coo.push(v, v + 1, -1.0)
      }
    }
  }
  coo.to_csr()
}

///|
/// A deterministic Fisher-Yates shuffle, so the example prints the same numbers
/// on every run and on every backend.
fn scramble_permutation(n : Int) -> Array[Int] {
  let perm : Array[Int] = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    perm[i] = i
  }
  let mut seed = 20260921
  for i = n - 1; i > 0; i = i - 1 {
    seed = (seed * 1103 + 12345) % 1000003
    let j = seed % (i + 1)
    let tmp = perm[i]
    perm[i] = perm[j]
    perm[j] = tmp
  }
  perm
}
