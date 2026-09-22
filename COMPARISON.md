# How this project relates to the existing MoonBit numerical code

This document exists because "there is already a sparse solver in MoonBit" is a
reasonable first reaction, and the answer has to be specific rather than
rhetorical. It records what was searched, what overlaps, and what does not.

## How the check was done, including where it failed

1. The registry index shipped with the toolchain (`$MOON_HOME/registry/index`)
   was flattened to the latest version of every module — **2,552 modules** — and
   searched for the vocabulary of this project.
2. Every module whose name, keywords or description matched matrices, linear
   algebra, solvers or factorisation was listed — **51 modules** — downloaded,
   and classified by the storage type in its generated interface rather than by
   what its description claims.
3. The public symbols this project exposes were grepped across those interfaces.

Two earlier passes of this check missed a module, and both misses were the same
mistake: **classifying a module by its name instead of reading its packages.**

- `xunyoyo/linalg` describes itself only as "Linear Algebra Library". It is a
  dense library over `Array[Array[Double]]`, so it does not overlap, but a
  reviewer found it before this document did.
- `hsy-bit/moonbit-circuit-solver` describes itself as a SPICE circuit
  simulator. Its `src/linalg` subpackage is a general sparse linear algebra
  implementation, and it does overlap. That is what the rest of this document is
  about.

The lesson is written down rather than assumed: a module's name is not evidence
about its contents, and the only reliable check is reading the generated
interface of every package inside it.

## The overlapping module, disclosed

`hsy-bit/moonbit-circuit-solver` version 0.1.1, package `src/linalg`, **4,793
lines of non-test code**. It is the numerical backend of a circuit simulator:
the simulator's `src/mna`, `src/nonlinear`, `src/transient` and `src/spice`
packages are what it exists to serve, and `src/linalg` is the linear algebra
they are built on.

Capability by capability, from its generated interface:

| Capability | `moonbit-circuit-solver/src/linalg` | this project |
|---|---|---|
| Triplet assembly | `SparseBuilder`, `MatrixTriplet` | `Coo`, with duplicate summing and a duplicate count |
| CSR storage | `CsrMatrix` | `Csr` |
| CSR public surface | 5 methods: `from_dense`, `to_dense`, `multiply_vector`, `nnz`, `get` | validated construction, binary-search lookup, `get_or`, `contains`, `row_nnz`, `row_range`, `max_row_nnz`, `diagonal`, `is_symmetric`, `spmv_add`, `scale`, `to_dense` |
| **CSC storage** | **none** | **`Csc`** |
| **Transposed product `A^T x`** | **none** — transpose exists only for dense `Matrix` and `ComplexMatrix` | **`Csc::spmv_transpose`** |
| **Sparse-dense product** | none | `spmm` |
| **Sparse-sparse product** | **none** | **`spgemm`** |
| Sparse addition and subtraction | none | `add`, `sub`, with exact cancellations dropped |
| Conjugate gradient | `conjugate_gradient_solve` | `cg` |
| BiCGSTAB | `bicgstab_solve` | `bicgstab` |
| GMRES | `gmres_solve`, `fgmres_solve` | `gmres`, restarted |
| Other Krylov methods | **TFQMR, MINRES** | none |
| **Warm starting** | **none** — no solver takes an initial guess | `initial?` on all three solvers |
| ILU(0) preconditioner | `Ilu0Preconditioner` | `Ilu0Precond` |
| Other preconditioners | **incomplete Cholesky, block Jacobi, SSOR** | none |
| Stationary iterations | **Jacobi, SOR** | none |
| Sparse Cholesky | `sparse_cholesky_solve` | `CholeskyFactor`, with `log_determinant` and a checked constructor |
| Sparse LU | **`factorize_sparse_lu`, with Markowitz pivoting** | none |
| Reverse Cuthill-McKee | `reverse_cuthill_mckee` | `reverse_cuthill_mckee` |
| Bandwidth | `compute_matrix_bandwidth` | `bandwidth`, plus `profile` |
| Matrix Market | `parse_matrix_market`, `serialize_matrix_market` | coordinate and array form, real, integer and pattern fields, general, symmetric and skew-symmetric symmetry; `write_matrix_market_symmetric` |
| **Graph layer** | **none** | **degree, adjacency construction, Laplacian, normalized Laplacian, connected components, PageRank** |
| **Scalar types** | **`Double` only** | **generic over `Add`/`Mul`/`Default`, with `Float`, `Int64` and two semirings** |
| Dense solvers, complex matrices, condition estimator | **yes** | none — out of scope |

The overlap is real and it is the solver core: CSR storage with a sparse
product, conjugate gradient, BiCGSTAB, GMRES, ILU(0), sparse Cholesky, reverse
Cuthill-McKee, bandwidth, and Matrix Market.

## What this project is, given that

**It is a general sparse linear algebra package, not the numerical backend of
one application.** That distinction is not a slogan; it is what the two
libraries' interfaces show. Their CSR type exposes five methods, all of them
sufficient for a circuit simulator and none of them sufficient for anything
else: there is no way to ask whether a matrix is symmetric, no way to get its
diagonal, no second storage format, and no matrix product other than
matrix-times-vector. A circuit simulator assembles a matrix, factorises it, and
solves; it never needs to multiply two sparse matrices or to store a matrix by
columns.

Four consequences, each of which is a capability the other library does not
have:

**1. Column storage and the transposed product.** `Csc` stores a matrix by
column, which makes `A^T x` a contiguous inner product instead of a scatter, and
makes column-oriented access a slice instead of a scan. The non-symmetric
solvers want `A^T x`, and so does anyone computing with a matrix whose rows are
the awkward direction.

**2. The sparse products.** `spgemm` and `spmm` are absent from the other
library entirely. They are what turns a graph's adjacency matrix into
reachability or shortest paths, and what lets an operator be applied to several
right-hand sides at once.

**3. Genericity, and what it buys.** The other library is `Double` throughout.
This one is generic over the scalar type, and the consequence is not
cosmetic — it is the difference between one algorithm and several:

```moonbit
// reachability, over the Boolean semiring where add is OR and multiply is AND
let closure = @ops.spgemm(adjacency, adjacency)

// shortest paths, over min-plus where add is min and multiply is plus
let distances = @ops.spgemm(weights, weights)
```

The same `spgemm` in both cases, with no change to the storage or the index
arithmetic. `examples/reachability` runs both and prints the results: on its
graph, the closure discovers that getting from `a` to `d` costs 4 by going
through `b` and `c`, against 9 for the direct edge. The `Int64` path keeps exact
integer assembly, which no floating point type can.

**4. The graph layer.** Laplacians, connected components and PageRank are not in
the other library. `examples/pagerank` ranks 500 generated pages with 2,474
stored entries where a dense matrix would hold 250,000.

## Correctness is checked against an independent implementation

`integration/` contains a dense reference implementation written separately from
the textbook definitions: one flat row-major array with no index arrays at all,
Gaussian elimination with partial pivoting, and an independently written
Cholesky. Thirteen randomised differential tests run every sparse routine
against it.

This matters because of how sparse code fails. A wrong `row_ptr` entry still
produces a number, the result is still lower triangular, and it still looks like
a solution — nothing announces that it is wrong. Hand-written expectations
verify the cases the author thought of; a second implementation written from a
different algorithm is what catches the rest. It found the Laplacian self-loop
bug during development, and it would have found the transpose bug that the
round-trip test could not see.

## What this project does not claim

- It does not claim to be the first sparse solver in MoonBit. It is not.
- It does not claim to replace `moonbit-circuit-solver/src/linalg`. That library
  has sparse LU with Markowitz pivoting, TFQMR, MINRES, incomplete Cholesky,
  block Jacobi and SSOR preconditioning, and SOR — none of which this project
  has, and a circuit simulator is better served by it.
- It does not claim the dense libraries are inadequate. They are the right tool
  for small dense matrices, which this project does not handle at all.
- Reverse Cuthill-McKee is a cheap heuristic here. Minimum-degree and
  nested-dissection orderings generally beat it, and neither is implemented.
