// Solving a 2D Poisson problem, and checking the answer against the exact one.
//
// The problem is the textbook one: on the unit square, find u with
//
//   -Laplacian(u) = f,   u = 0 on the boundary,
//   f(x, y) = 2 pi^2 sin(pi x) sin(pi y),
//
// whose exact solution is u(x, y) = sin(pi x) sin(pi y). Discretising on a grid
// with the 5-point stencil turns it into a sparse linear system, and because
// the exact solution is known, the error can be measured rather than only the
// residual.
//
// Run from the project root:
//
//   moon run examples/poisson

///|
/// The full grid is `GRID x GRID` with the boundary pinned, leaving
/// `(GRID - 2)^2` unknowns.
const GRID : Int = 100

///|
fn main raise {
  let n = GRID
  let m = n - 2
  let h = 1.0 / (n - 1).to_double()
  let h2 = h * h
  let unknowns = m * m
  println("2D Poisson on a \{n}x\{n} grid")
  println("  unknowns             : \{unknowns}")
  println("  grid spacing         : \{h}")

  // Assemble the 5-point stencil. Each interior point couples to its four
  // neighbours and to itself.
  let coo : @coo.Coo[Double] = @coo.Coo::empty(unknowns, unknowns)
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      let row = i * m + j
      coo.push(row, row, 4.0 / h2)
      if i > 0 {
        coo.push(row, (i - 1) * m + j, -1.0 / h2)
      }
      if i + 1 < m {
        coo.push(row, (i + 1) * m + j, -1.0 / h2)
      }
      if j > 0 {
        coo.push(row, i * m + (j - 1), -1.0 / h2)
      }
      if j + 1 < m {
        coo.push(row, i * m + (j + 1), -1.0 / h2)
      }
    }
  }
  let a = coo.to_csr()

  // The right-hand side, and the exact solution, sampled at the interior points.
  let b : Array[Double] = Array::make(unknowns, 0.0)
  let exact : Array[Double] = Array::make(unknowns, 0.0)
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < m; j = j + 1 {
      let x = (i + 1).to_double() * h
      let y = (j + 1).to_double() * h
      let s = @math.sin(@math.PI * x) * @math.sin(@math.PI * y)
      b[i * m + j] = 2.0 * @math.PI * @math.PI * s
      exact[i * m + j] = s
    }
  }

  let dense_gib = unknowns.to_double() *
    unknowns.to_double() *
    8.0 /
    1073741824.0
  let density = a.nnz().to_double() /
    (unknowns.to_double() * unknowns.to_double()) *
    100.0
  println("  stored entries       : \{a.nnz()}  (density \{density}%)")
  println("  dense would need     : \{dense_gib} GiB for the values alone")
  println("")

  // Conjugate gradient: one product with the matrix per iteration.
  let opts = @solve.SolverOptions::new(1.0e-10, 5000)
  let cg_result = @solve.cg(
    a,
    b,
    options=opts,
    precond=@solve.Precond::Ilu0(@solve.Ilu0Precond::new(a)),
  )
  println("conjugate gradient with ILU(0) preconditioning")
  println("  converged            : \{cg_result.converged}")
  println("  iterations           : \{cg_result.iterations}")
  println("  relative residual    : \{cg_result.residual}")
  println("  max error vs exact   : \{max_abs_error(cg_result.x, exact)}")
  println("")

  // A direct factorisation: exact up to rounding, but it fills in.
  let factor = @solve.CholeskyFactor::new(a)
  let x_direct = factor.solve(b)
  let fill_ratio = factor.nnz().to_double() / a.nnz().to_double()
  println("sparse cholesky")
  println("  entries in matrix    : \{a.nnz()}")
  println(
    "  entries in factor    : \{factor.nnz()}  (\{fill_ratio}x the matrix)",
  )
  println("  max error vs exact   : \{max_abs_error(x_direct, exact)}")
  println("")
  println("the two solutions should agree with each other:")
  println("  max difference       : \{max_abs_error(cg_result.x, x_direct)}")
  println("")
  println("see examples/reordering for what reordering does to the fill-in")
}

///|
/// Largest absolute difference between two vectors.
fn max_abs_error(got : Array[Double], want : Array[Double]) -> Double {
  let mut worst = 0.0
  for i = 0; i < got.length(); i = i + 1 {
    let d = Double::abs(got[i] - want[i])
    if d > worst {
      worst = d
    }
  }
  worst
}
