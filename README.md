# sparse-linalg

[![mooncakes.io](https://img.shields.io/badge/mooncakes.io-0.1.0-blue)](https://mooncakes.io/docs/lhw153766-ctrl/sparse-linalg)
[![CI](https://github.com/lhw153766-ctrl/sparse-linalg/actions/workflows/ci.yml/badge.svg)](https://github.com/lhw153766-ctrl/sparse-linalg/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Numeric sparse linear algebra for MoonBit: compressed storage, sparse kernels,
and solvers that work on matrices too large to store densely.

Published on mooncakes.io as
[`lhw153766-ctrl/sparse-linalg@0.1.0`](https://mooncakes.io/docs/lhw153766-ctrl/sparse-linalg).

A 100,000 x 100,000 matrix with about ten non-zeros per row has 10<sup>10</sup>
dense entries — roughly 80 GB as `Double`, which no machine you are likely to
be sitting at will allocate. The same matrix in compressed sparse row form
holds about 10<sup>6</sup> entries, a few megabytes. Every routine here works on
that second representation.

**Status:** early. The storage formats, kernels, solvers and graph algorithms
are in place; the API may still change before 0.2.0.

## What is here

| Package | Contents |
|---|---|
| `types` | `SparseError`, `Shape`, index validation |
| `coo` | Triplet storage and assembly, summing duplicate coordinates |
| `csr` | Compressed sparse row: `y = A x`, `y = alpha A x + beta y`, diagonal, symmetry check |
| `csc` | Compressed sparse column: `y = A^T x`, `O(nnz)` conversions |
| `ops` | Addition, subtraction, transposition, sparse-dense and sparse-sparse products |
| `solve` | Sparse Cholesky, conjugate gradient, BiCGSTAB, GMRES, Jacobi and ILU(0) preconditioners |
| `reorder` | Reverse Cuthill-McKee, bandwidth and profile measurement |
| `io` | Matrix Market (`.mtx`) reading and writing |
| `graph` | Degree, adjacency construction, Laplacians, connected components, PageRank |
| `semiring` | `BoolSemiring` and `MinPlus`, which turn the same product into reachability or shortest paths |

4,900 lines of MoonBit, plus 3,600 lines of tests.

## Install

Add the dependency to your `moon.mod`:

```
import {
  "lhw153766-ctrl/sparse-linalg@0.1.0",
}
```

Declare the packages you use in your `moon.pkg`:

```
import {
  "lhw153766-ctrl/sparse-linalg/coo" @coo,
  "lhw153766-ctrl/sparse-linalg/solve" @solve,
}
```

## Usage

### Assemble from coordinates

Real matrices arrive as a list of `(row, col, value)` triples. `Coo` accepts
them in any order and allows repeats, summing duplicates during conversion — so
a finite-element assembly loop can simply append without tracking which
coordinates it has already touched.

```moonbit
let a : @coo.Coo[Double] = @coo.Coo::empty(3, 3)
a.push(0, 0, 2.0)
a.push(1, 1, 3.0)
a.push(2, 2, 4.0)
a.push(0, 1, -1.0)
a.push(1, 0, -1.0)   // the mirror of the previous entry
let csr = a.to_csr()
```

### Multiply

```moonbit
let y = csr.spmv([1.0, 1.0, 1.0])
```

`spmv` touches each stored entry once, so its cost is `O(nnz)` regardless of how
many zeros the dense form would have held.

### Solve

Iterative solvers need one product per iteration and take a preconditioner:

```moonbit
let opts = @solve.SolverOptions::new(1.0e-10, 5000)
let result = @solve.cg(a, b, options=opts, precond=@solve.Precond::Ilu0(@solve.Ilu0Precond::new(a)))
println(result.converged)   // true
println(result.iterations)  // 80
println(result.residual)    // 7.5e-11
```

`converged` is reported separately from `iterations`, so "solved" is
distinguishable from "ran out of budget"; a result that did not converge still
holds the best iterate reached.

Direct factorisation, when the matrix is symmetric positive definite:

```moonbit
let factor = @solve.CholeskyFactor::new_checked(a)
let x = factor.solve(b)
println(factor.log_determinant())
```

`new_checked` verifies symmetry first. `new` skips that check, which is what a
caller holding only one triangle wants and what a caller who is not sure does
not.

### Reorder before factorising

A sparse factor fills in, and how much depends on the elimination order:

```moonbit
let (ordered, perm) = @reorder.reorder_rcm(a)
let factor = @solve.CholeskyFactor::new(ordered)
```

The permutation uses the convention `perm[old_index] = new_index`, so a
right-hand side has to be permuted the same way before it can be used with the
reordered matrix.

### Read a Matrix Market file

```moonbit
let file = @io.parse_matrix_market_full(text)
println(file.header.symmetry)     // symmetric
println(file.matrix.nnz())        // stored triples, after mirroring
let a = file.matrix.to_csr()
```

The package works on strings rather than files, so it keeps working on backends
without a filesystem.

### Graphs

```moonbit
let adjacency = @graph.adjacency_from_edges(n, [(0, 1), (1, 2)])
let laplacian = @graph.laplacian(adjacency)
let ranks = @graph.page_rank(link_matrix)
```

## Examples

Three runnable programs, each printing what it computed so the output can be
checked by eye:

```
moon run examples/poisson
moon run examples/reordering
moon run examples/reachability
moon run examples/pagerank
moon run examples/matrix_market
```

`examples/poisson` solves the 2D Poisson equation on a 100x100 grid (9,604
unknowns, 47,628 stored entries, 0.05% dense) and checks the answer against the
closed-form solution. Conjugate gradient with ILU(0) converges in 80 iterations
to a relative residual of 7.5e-11; the error against the exact solution is
8.4e-5, which is the `O(h^2)` discretisation error for this grid spacing rather
than a solver error, and the direct factorisation agrees with it to 3.7e-12.

`examples/reordering` measures what reordering buys, on a 60x60 grid Laplacian
under three numberings of the same matrix:

| numbering | bandwidth | factor entries |
|---|---|---|
| natural (row-major) | 60 | 216,059 |
| scrambled | 3,558 | 1,072,317 |
| scrambled, then RCM | 60 | 149,330 |

The natural ordering is the baseline on purpose: for a grid it is already good,
and it would be easy to claim an improvement that is not there. RCM recovers
the bandwidth of the scrambled matrix exactly and ends up with a factor smaller
than the natural numbering's, without being told the matrix is a grid.

## Performance

`moon bench` measures the kernels. On a tridiagonal system, ten times the
stored entries takes ten times as long — which is the property that matters,
since the dense counterpart of the 10000-point case would hold 100,000,000
entries:

| kernel | time |
|---|---|
| `spmv` CSR, 1000x1000, 2,998 stored entries | 12.95 µs |
| `spmv` CSC, 1000x1000 | 14.53 µs |
| `spmv_transpose` CSC, 1000x1000 | 12.91 µs |
| `spmv_add`, 1000x1000 | 12.46 µs |
| `spmv` CSR, 10000x10000, 29,998 stored entries | 129.32 µs |

## Warm starting

A solver can start from a previous solution, which is what a time-stepping loop
wants:

```moonbit
let result = @solve.cg(a, b, initial=Some(previous_solution))
```

On a 20x20 grid Laplacian, starting from a solution perturbed by 1e-6 takes 26
iterations against 40 from zero. It does not always pay — on a 1D tridiagonal
system with a constant right-hand side the iteration count is the same either
way, because there the convergence is limited by the smoothness of the
right-hand side rather than by the size of the starting residual.

## Reproduce the test suite

```
moon update
moon check --target all
moon build --target all
moon test  --target all
moon fmt --check
```

213 tests, passing on `wasm`, `wasm-gc`, `js` and `native`. CI runs the same
commands on Linux, macOS and Windows.

Three kinds of test are used, and the distinction matters:

- **Hand-checked cases** in each package's `*_test.mbt`, where the expected
  value can be derived on paper. The sparse Cholesky tests, for instance, assert
  the exact factor structure of the 2x2 grid stencil, which is short enough to
  work out by hand.
- **Structural invariants**: `L L^T` reconstructs `A`, a permutation really is
  a permutation, `A` and its transpose have the same stored values.
- **Differential tests** in `integration/`, which run every sparse routine
  against an independent dense implementation written from the textbook
  definitions — flat row-major arrays, Gaussian elimination with partial
  pivoting, a different Cholesky. This is the check that catches the failure
  mode the others cannot: a wrong `row_ptr` entry still produces a plausible
  number, so only a second implementation disagrees visibly. The two share no
  code path.

`moon check --deny-warn` is clean.

The published artifact was checked separately from the working tree: version
0.1.0 was installed into a fresh project from the registry, and a test there
resolved the imports through the download and solved a system with both the
direct and the iterative solver. Presence in the registry is not the same as
being installable.

## One product, several algorithms

The storage and the products are generic over the scalar type, which is not a
cosmetic choice: it is what lets the same `spgemm` compute different things.

```moonbit
// Reachability: over the Boolean semiring, add is OR and multiply is AND.
let closure = @ops.spgemm(adjacency, adjacency)

// Shortest paths: over min-plus, add is min and multiply is plus.
let distances = @ops.spgemm(weights, weights)
```

`examples/reachability` runs both over the same code and prints the results. On
its graph the closure finds that going from `a` to `d` costs 4 via `b` and `c`,
against 9 for the direct edge. The `Int64` path keeps exact integer assembly.

## How this relates to the existing MoonBit sparse code

There is an existing sparse implementation in the ecosystem:
`hsy-bit/moonbit-circuit-solver/src/linalg`, the numerical backend of a SPICE
circuit simulator. It overlaps with this project in the solver core — CSR
storage and the sparse product, conjugate gradient, BiCGSTAB, GMRES, ILU(0),
sparse Cholesky, reverse Cuthill-McKee, and Matrix Market.

The boundary is set out capability by capability in
[COMPARISON.md](COMPARISON.md). In short: this project adds column storage and
the transposed product, the sparse-sparse and sparse-dense products, genericity
over the scalar type with two semirings, and a graph layer — none of which the
other library has. The other library has sparse LU with Markowitz pivoting,
TFQMR, MINRES, incomplete Cholesky, block Jacobi and SSOR preconditioning, and
SOR, none of which this project has.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Development log

[docs/development-log.md](docs/development-log.md) records the design decisions
and trade-offs, the six bugs that were found while building this and how each
one was caught, the verification strategy, and the known limitations — including
the ones that are still sharp edges.

## How this relates to the other MoonBit numerical libraries

Short version: the existing libraries are dense and this one is sparse. The
long version, with the registry-wide search behind it, is in
[COMPARISON.md](COMPARISON.md).
