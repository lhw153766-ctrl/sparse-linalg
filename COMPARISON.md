# How this differs from the existing MoonBit numerical libraries

This document exists because "there is already a linear algebra library for
MoonBit" is a reasonable first reaction, and it is worth answering with
specifics rather than with a claim of novelty.

## How the comparison was made

The check was done against the full published registry, not against a keyword
search:

1. The registry index shipped with the toolchain
   (`$MOON_HOME/registry/index`) was flattened to the latest version of every
   module — 2,552 modules — into a table of name, version, keywords and
   description. That table was then searched for the vocabulary of this project
   (`sparse matrix`, `csr`, `csc`, `krylov`, `conjugate gradient`, `gmres`,
   `preconditioner`, `suitesparse`, `reverse cuthill`, and others).
2. Every numerically-oriented module found that way was downloaded and its
   generated interfaces (`pkg.generated.mbti`) read directly, including root
   packages. Reading interfaces rather than package names matters: a module can
   publish many packages and the root package is often the one holding the real
   functionality.
3. The 20 public symbols this project intends to expose were grepped across
   those interfaces.

## The three libraries a reviewer will find first

| | this project | `Luna-Flow/linear-algebra` | `amor2025/moonNum` | `CAIMEOX/symbit` |
|---|---|---|---|---|
| Storage | `Coo` / `Csr` / `Csc`, three parallel arrays | `NdArray`, contiguous dense memory | `NdArray`, dense tensor | `Expr` syntax tree |
| Space for an `n x n` matrix | `O(nnz)` | `O(n^2)` | `O(n^2)` | `O(nnz)` of symbolic nodes |
| Element type | any scalar with `Add`/`Mul`/`Default`, exact types included | numeric field | numeric | symbolic expressions |
| Solving | iterative Krylov (`cg`, `bicgstab`, `gmres`) and sparse Cholesky | dense `cholesky_decomposition`, `eigen` | dense `cholesky`, `solve`, `svd` | symbolic elimination, exact rationals |
| Typical use | 10^5-node graph Laplacians, PDE meshes, sparse least squares | small dense graphics/ML matrices | dense NumPy-style array work | symbolic algebra, calculus, theorem work |

The upstream project's own curated listing describes `Luna-Flow/linear-algebra`
as *"Checked matrix and vector APIs with mutable and immutable **dense**
representations"* — dense is the intended scope there, not a gap someone forgot
to fill.

Concretely: for a 100,000 x 100,000 matrix with about ten non-zeros per row,
the dense libraries need on the order of 80 GB for the values alone and cannot
be used at all, while the CSR form needs a few megabytes and `spmv` finishes in
milliseconds. This is not a performance difference, it is the difference between
running and not running.

## Capability by capability

| Capability | Already present | What this project adds | Evidence |
|---|---|---|---|
| Numeric sparse storage | nothing | `Coo`, `Csr`, `Csc` with validated construction | registry search for `sparse matrix` / `csr` / `csc` returns no numeric implementation; `Luna-Flow/linear-algebra/mutable` exposes only dense `Matrix` |
| Symbolic sparse matrices | `CAIMEOX/symbit` `SparseMatrix`, `to_sparse` | — (deliberately not duplicated) | `symbit/src/symmatrices/pkg.generated.mbti` shows `SparseMatrix` over `Expr`, a computer-algebra type, not a numeric one |
| `y = A x` for sparse `A` | nothing | `Csr::spmv`, `Csc::spmv` | no `spmv` / `spmm` symbol in any downloaded interface |
| `y = A^T x` | nothing | `Csc::spmv_transpose` | no transposed sparse product anywhere |
| Sparse direct factorisation | dense only | sparse Cholesky | `amor2025/moonNum/src/linalg` has `cholesky(NdArray)`; dense input only |
| Krylov iterative solvers | nothing | `cg`, `bicgstab`, `gmres` | registry search for `krylov`, `conjugate gradient`, `gmres` returns zero results |
| Preconditioners | nothing | Jacobi, ILU(0) | no `preconditioner` / `ilu` symbol in any interface |
| Bandwidth-reducing reordering | nothing | reverse Cuthill-McKee | no `reverse_cuthill_mckee` / `reorder` symbol in any numerical interface |
| Matrix Market I/O | nothing | reader and writer | no `matrix_market` symbol in any interface |
| Dense solvers | `moonNum`, `Luna-Flow/linear-algebra` | — (used as a test oracle instead) | see below |

## What this project does not claim

- It does not claim to replace the dense libraries. Dense small matrices are
  handled better by contiguous storage, and this project has no dense
  factorisation to offer.
- It does not claim to be the first MoonBit matrix library, only the first with
  numeric sparse storage and sparse solvers.
- Sparse *symbolic* matrices already exist in `symbit`. That is a different
  problem — exact manipulation of expressions rather than floating-point
  computation — and this project does not attempt it.

## The dense libraries are used as the correctness oracle

Rather than only comparing against hand-written expectations, the test suite
solves the same systems twice: once through the sparse path, and once by
expanding to dense and calling an independent implementation from
`amor2025/moonNum`. The two must agree to a tolerance.

This makes the relationship complementary rather than competitive. A bug in the
sparse index arithmetic — the failure mode that matters here, since a wrong
`row_ptr` entry still yields a plausible number — shows up as a disagreement
with a separately written library.
