// An independent dense reference implementation.
//
// These routines exist to be a second opinion. They use a different data
// layout (one flat row-major array, no index arrays at all) and different
// algorithms (Gaussian elimination with partial pivoting rather than a
// simplicial sparse Cholesky), so they share no code path with the sparse
// implementations they check.
//
// That matters because the failure mode of sparse index arithmetic is silent:
// a wrong `row_ptr` entry still produces a plausible number. Comparing against
// a dense implementation that never computes a row pointer is what turns that
// silent wrongness into a test failure.
//
// They are deliberately naive — `O(n^3)`, no blocking, no pivoting strategy
// beyond partial pivoting — because they only ever run on small systems.

///|
/// Dense matrix-vector product, row-major.
pub fn dense_matvec(
  a : Array[Double],
  n : Int,
  x : Array[Double],
) -> Array[Double] {
  let y : Array[Double] = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    let mut acc = 0.0
    for j = 0; j < n; j = j + 1 {
      acc = acc + a[i * n + j] * x[j]
    }
    y[i] = acc
  }
  y
}

///|
/// Dense matrix-matrix product, row-major.
pub fn dense_matmul(
  a : Array[Double],
  b : Array[Double],
  n : Int,
) -> Array[Double] {
  let c : Array[Double] = Array::make(n * n, 0.0)
  for i = 0; i < n; i = i + 1 {
    for k = 0; k < n; k = k + 1 {
      let aik = a[i * n + k]
      for j = 0; j < n; j = j + 1 {
        c[i * n + j] = c[i * n + j] + aik * b[k * n + j]
      }
    }
  }
  c
}

///|
/// Solve `A x = b` by Gaussian elimination with partial pivoting.
///
/// Returns `None` when a pivot is too small to divide by, which is how a
/// singular system is reported.
pub fn dense_solve(
  a_in : Array[Double],
  b_in : Array[Double],
  n : Int,
) -> Array[Double]? {
  let a = a_in.copy()
  let x = b_in.copy()
  for col = 0; col < n; col = col + 1 {
    // Partial pivoting: take the largest magnitude entry in this column.
    let mut pivot_row = col
    let mut best = Double::abs(a[col * n + col])
    for r = col + 1; r < n; r = r + 1 {
      let v = Double::abs(a[r * n + col])
      if v > best {
        best = v
        pivot_row = r
      }
    }
    if best < 1.0e-14 {
      return None
    }
    if pivot_row != col {
      for j = 0; j < n; j = j + 1 {
        let tmp = a[col * n + j]
        a[col * n + j] = a[pivot_row * n + j]
        a[pivot_row * n + j] = tmp
      }
      let tmp = x[col]
      x[col] = x[pivot_row]
      x[pivot_row] = tmp
    }
    let pivot = a[col * n + col]
    for r = col + 1; r < n; r = r + 1 {
      let factor = a[r * n + col] / pivot
      if factor != 0.0 {
        for j = col; j < n; j = j + 1 {
          a[r * n + j] = a[r * n + j] - factor * a[col * n + j]
        }
        x[r] = x[r] - factor * x[col]
      }
    }
  }
  // Back substitution.
  for ii = 0; ii < n; ii = ii + 1 {
    let i = n - 1 - ii
    let mut acc = x[i]
    for j = i + 1; j < n; j = j + 1 {
      acc = acc - a[i * n + j] * x[j]
    }
    x[i] = acc / a[i * n + i]
  }
  Some(x)
}

///|
/// Dense Cholesky, returning the lower triangle in a full row-major array.
///
/// Returns `None` when the matrix is not positive definite.
pub fn dense_cholesky(a : Array[Double], n : Int) -> Array[Double]? {
  let l : Array[Double] = Array::make(n * n, 0.0)
  for i = 0; i < n; i = i + 1 {
    for j = 0; j <= i; j = j + 1 {
      let mut acc = a[i * n + j]
      for k = 0; k < j; k = k + 1 {
        acc = acc - l[i * n + k] * l[j * n + k]
      }
      if i == j {
        if !(acc > 0.0) {
          return None
        }
        l[i * n + i] = Double::sqrt(acc)
      } else {
        l[i * n + j] = acc / l[j * n + j]
      }
    }
  }
  Some(l)
}

///|
/// A small deterministic pseudo-random generator (the MINSTD Lehmer
/// generator).
///
/// Tests that generate random systems have to be reproducible, so this uses
/// fixed constants rather than anything seeded from the environment. The
/// multiply would overflow a 32-bit `Int`, so the state is carried in `Int64`.
pub struct Rng {
  mut state : Int64
}

///|
/// Start from a fixed seed.
pub fn Rng::new(seed : Int) -> Rng {
  { state: seed.to_int64() % 2147483647L, }
}

///|
/// The next raw value.
pub fn Rng::next(self : Rng) -> Int {
  self.state = self.state * 48271L % 2147483647L
  self.state.to_int()
}

///|
/// A uniform value in `[lo, hi)`.
pub fn Rng::uniform(self : Rng, lo : Double, hi : Double) -> Double {
  let r = self.next().to_double() / 2147483647.0
  lo + r * (hi - lo)
}

///|
/// An integer in `[0, n)`.
pub fn Rng::below(self : Rng, n : Int) -> Int {
  self.next() % n
}
