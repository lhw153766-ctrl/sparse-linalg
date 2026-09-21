# sparse-linalg

Numeric sparse linear algebra for MoonBit: compressed storage, sparse kernels,
and solvers that work on matrices too large to store densely.

A matrix with 100,000 rows and 100,000 columns whose rows hold about ten
non-zeros each has 10<sup>10</sup> dense entries — roughly 80 GB as `Double`,
which no machine you are likely to be sitting at will allocate. The same matrix
in CSR form holds about 10<sup>6</sup> entries, a few megabytes. Every routine
here works on that second representation.

**Status:** early. The storage formats, sparse kernels and the first solvers are
in place; the API may still change before 0.2.0.

## What is here

| Package | Contents |
|---|---|
| `types` | `SparseError`, `Shape`, index validation |
| `coo` | Triplet storage, assembly, duplicate summing |
| `csr` | Compressed sparse row, `y = A x`, `y = alpha A x + beta y` |
| `csc` | Compressed sparse column, `y = A^T x`, conversions |
| `ops` | Matrix addition, scaling, transpose, sparse-dense products |
| `io` | Matrix Market (`.mtx`) reader and writer |
| `solve` | Cholesky, conjugate gradient, BiCGSTAB, GMRES, preconditioners |
| `reorder` | Reverse Cuthill-McKee bandwidth reduction |
| `graph` | Graph Laplacian, PageRank, connected components |

## Install

Add the dependency to your `moon.mod`:

```
import {
  "lhw153766-ctrl/sparse-linalg@0.1.0",
}
```

Then declare the packages you use in your `moon.pkg`:

```
import {
  "lhw153766-ctrl/sparse-linalg/csr" @csr,
  "lhw153766-ctrl/sparse-linalg/solve" @solve,
}
```

## Usage

### Assemble from coordinates

Real matrices arrive as a list of `(row, col, value)` triples. `Coo` accepts
them in any order, allows repeats, and sums duplicates during conversion — so a
finite-element assembly loop can simply append without tracking which
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

```moonbit
// Conjugate gradient for a symmetric positive definite system.
let result = @solve.cg(csr, rhs, tol=1.0e-10, max_iter=1000)
```

Solvers return an iteration count and the achieved residual alongside the
solution, so a caller can tell "solved" from "ran out of budget" without
guessing.

### Read a Matrix Market file

```moonbit
let text = @fs.read_file("bcsstk01.mtx")
let a = @io.parse_matrix_market(text)
```

## Examples

Runnable programs live in `examples/`. Each one prints what it computed so the
output can be checked by eye:

```
moon run examples/poisson
moon run examples/pagerank
```

## Reproduce the test suite

```
moon update
moon check --target all
moon test  --target all
```

All 47 tests pass on `wasm`, `wasm-gc`, `js` and `native`. CI runs the same
commands on Linux, macOS and Windows.

Two kinds of test are used, and the distinction matters:

- **Hand-checked cases** in each package's `*_test.mbt`, where the expected
  value can be verified on paper.
- **Differential tests** against a dense reference. The same system is solved
  through the sparse path and through an independent dense implementation; the
  results must agree to a tolerance. This is the check that actually catches
  index-arithmetic bugs, because a wrong `row_ptr` entry still produces a
  plausible-looking number.

## License

Apache-2.0. See [LICENSE](LICENSE).
