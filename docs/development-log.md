# Development log

An account of how this library was built: the decisions, the trade-offs, the
bugs that were found and — more usefully — how each one was found, and what is
not done. Written to be read by someone who has to judge whether the code can
be trusted, so the failures are in here alongside the successes.

## Scope

The library is numeric sparse linear algebra. That boundary was chosen because
it is the one thing the MoonBit ecosystem does not already have: the existing
numerical libraries are dense, and the one sparse matrix type that exists
(`CAIMEOX/symbit`) holds symbolic expressions, not floating-point numbers.
`COMPARISON.md` records the registry-wide search behind that claim.

The scope was deliberately kept to algorithms that are *specified* rather than
*invented*: storage formats, the sparse products, four solvers, two
preconditioners, one reordering heuristic, one file format. Every one of them
has a textbook description and a known-correct answer, which is what makes the
verification below possible.

What was left out, and why:

- **No dense linear algebra.** A small dense matrix is better served by
  contiguous storage, and the existing libraries already do that well. Adding a
  dense path would have meant maintaining two code paths for one job.
- **No minimum-degree or nested-dissection ordering.** Both generally beat
  reverse Cuthill-McKee, and both are considerably more code. RCM is the honest
  entry point: cheap, well defined, and it reduces the fill on the cases where
  numbering is arbitrary.
- **No sparse LU with partial pivoting.** Sparse Cholesky covers the symmetric
  positive definite case, and the Krylov methods cover the rest. A sparse LU
  needs dynamic data structures and a pivoting strategy that interacts with the
  fill, which is a project in itself.
- **No complex scalars.** The Matrix Market reader refuses complex input rather
  than silently keeping the real part. Accepting a matrix and quietly discarding
  half of it would be worse than an error.

## Design decisions

**Storage and kernels are generic; the solvers are `Double`.** The formats and
the products are generic over any scalar with `Add`, `Mul` and `Default`, so
`Double`, `Float` and `Int64` all work — the tests include an `Int64` case that
asserts `9000000000000 + 7` is not swallowed by floating point. The solvers are
not generic, because every one of them needs a convergence test of the form "is
this residual small enough", which only means something for floating point.
Making them generic would have meant pretending otherwise.

**A custom scalar trait was tried and abandoned.** The first attempt defined a
`SparseScalar` trait carrying the arithmetic plus conversions to and from
`Double`, so the solvers could be generic too. Two problems killed it: this
toolchain's `impl` blocks promote method names to package-level identifiers, so
implementing the same trait for `Double` and `Float` in one package collides;
and implicit method promotion for built-in types is deprecated, so
`(2.0).to_f64()` does not resolve. The design that survived uses the core traits
for storage and arithmetic, and restricts the solver layer to `Double`.

**Both CSR and CSC, not one of them.** CSC earns its place because `A^T x` is a
plain product in CSC and needs a transpose in CSR, and the non-symmetric solvers
want `A^T x`. The conversion between the two is a counting sort — `O(nnz)`, no
comparison sort — because the destination order is determined by the pointer
array rather than by ordering values.

**Validation at the boundary, not in the inner loop.** `Csr::new` checks every
structural invariant: pointer array length, monotonicity, that it starts at zero
and ends at `nnz`, that column indices are in range and strictly increasing
within a row. `Csr::new_unchecked` skips all of it for builders that have just
produced valid arrays. The cost is paid once at construction, and a malformed
matrix is rejected where it is built rather than producing plausible wrong
numbers a hundred lines later.

**I/O works on strings, not files.** The Matrix Market package takes and returns
`String`. This keeps it working on `wasm` and `wasm-gc`, which have no
filesystem, and it is why the whole test suite runs on all four backends. The
cost is that a caller with a file has to read it themselves.

**Preconditioners are an enum, not a trait.** A trait would let callers add
their own, but this toolchain's trait dispatch and method-promotion rules made
that awkward, and an enum keeps every built-in variant visible in the generated
interface.

**Solvers report `converged` separately from `iterations`.** A result that ran
out of budget still holds the best iterate reached, and a caller needs to be
able to tell "solved" from "gave up" without inferring it from a residual.

## Bugs found, and how

These are the interesting part. Each one is a case where the code was wrong and
the tests that existed at the time did not notice.

**The transpose did not transpose.** `ops::transpose` converted CSR to CSC and
back. That is a re-layout, not a transpose: the shape came back unchanged. The
round-trip test passed, the symmetric-matrix test passed, and the `transpose ∘
transpose = identity` test passed — all three are satisfied by a function that
does nothing. Only the rectangular test caught it, because only a rectangular
matrix has a shape that can be wrong. The fix was to reinterpret the row arrays
as column arrays, which *is* the transpose, and the test suite gained an
entry-by-entry check on a non-symmetric rectangular matrix plus a cross-check
that `transpose(A).spmv(x)` equals `Csc::spmv_transpose(x)`.

**The CSR/CSC conversion crashed on an empty matrix.** The scatter sized its
output arrays from `vals[0]`, which does not exist when there are no stored
entries. Every test used a non-empty matrix, so nothing failed until a transpose
of an empty matrix was tried. The fix is an early return for the empty case.

**The Laplacian counted self-loops.** `L = D - A` is conventionally defined so
that a self-loop cancels out, leaving the row sums at zero. The implementation
added the self-loop to the diagonal instead, so a graph with a self-loop had row
sums equal to the loop weight. The hand-written tests all used loop-free graphs.
The differential test — which builds `D - A` densely from the same adjacency
matrix — failed on a random graph that happened to contain one.

**The permutation convention was inconsistent.** `permute` implemented
`B[perm[i]][perm[j]] = A[i][j]` while `reverse_cuthill_mckee` returned the
inverse of that, so RCM was applied as its own inverse. The result was a valid
permutation, a symmetric matrix, an unchanged number of stored entries, and a
*bandwidth that went up instead of down*. The bug was invisible to every
structural check; it took measuring the bandwidth and finding it worse than the
input. Both functions now use `perm[old] = new`, documented at both sites, and
the tests assert that the bandwidth actually falls.

**The reordering tests reused an unpermuted right-hand side.** Solving the
reordered system with the original `b` is wrong: row `perm[i]` of the reordered
matrix corresponds to row `i` of the original and must carry `b[i]`. The test
passed anyway, because the vector it used happened to be constant. It was only
caught when the same scenario was written with a random right-hand side in the
differential suite. The reordering test now uses a non-constant `b` so it cannot
pass by accident.

**The Matrix Market comment filter also removed the banner.** The banner line
begins with `%%`, like a comment, so filtering comment lines dropped it and the
parser then read the size line as a header. Every coordinate-format test failed
with "header needs five fields". The banner is now taken off the front before
comments are dropped.

**The closure algorithm in my own test was wrong.** The first version of the
semiring tests accumulated the powers `A, A^2, A^4, A^8` and called that a
transitive closure. It is not: it covers only path lengths that are powers of
two, so the three-step cycle `a -> b -> c -> a` is missed and vertex `a` appears
not to reach itself. Squaring the *accumulated* relation is what works. The
library was right and the test was wrong, and the corrected test now says so in
a comment.

**Two test expectations were wrong, not the code.** Worth recording because it
cuts the other way: on the fill-in test the expected count of 13 was wrong
because the comparison was against the wrong baseline — `A` stores both
triangles and `L` only one, so the right baseline is 8, and the measured 9 is
exactly one fill entry. And on the preconditioned CG test, the assertion that
plain CG would converge was wrong: it stalls at 2.2e-10, which is the double
precision floor for that condition number. Both tests were rewritten to assert
what is actually true, and the second one became a better test — it now asserts
that Jacobi reaches a residual plain CG cannot.

## Toolchain findings worth writing down

Three things about this toolchain cost real time and are not documented in the
places I looked. They are recorded here because the next person will hit them.

- **A trait implementation needs an explicit `pub` to be visible outside its own
  package.** Without it the implementation works inside the defining package and
  fails to resolve everywhere else, and the error says only that the type does
  not implement the trait. I lost an hour to this before finding `pub impl` in a
  generated interface.
- **A tuple struct does not pick up a core trait implementation.** `struct W(Double)`
  with `impl Add for W` does not resolve, while `struct W { v : Double }` does.
- **A struct literal needs a trailing comma or it parses as a block.**
  `{ v }` is ambiguous; `{ v, }` is a struct.

Two of these were found by building the semiring package, which exists partly to
make the genericity claim testable. That was worth doing for its own sake: the
claim would have been false as first written, and only running it showed that.

## Verification strategy

Hand-checked expectations are necessary and not sufficient. They verify the
cases the author thought of. For sparse index arithmetic specifically, there is
a worse problem: **the failure mode is silent**. A wrong `row_ptr` entry still
produces a number, it is still lower triangular, it still looks like a solution.
Nothing about the output announces that it is wrong.

So `integration/` contains a second implementation, written from the textbook
definitions and sharing no code path with the first: one flat row-major array
with no index arrays at all, Gaussian elimination with partial pivoting, and an
independently written Cholesky. Thirteen randomised differential tests run every
sparse routine against it. Random inputs come from a deterministic generator, so
a failure is reproducible.

That arrangement found the Laplacian self-loop bug, and it would have found the
transpose bug immediately.

The plan had been to use an existing library as the external oracle, which is a
stronger claim than a second implementation by the same author. That was
dropped: the candidate, `amor2025/moonNum`, pulls in `Kaida-Amethyst/openblas`
and then `Kaida-Amethyst/python`, whose prebuild script runs `python3` and broke
`moon test` locally. Shipping a dependency that breaks the test command to get a
nicer verification story is a bad trade, and `COMPARISON.md` records the
decision.

## Use of AI

The code, tests and documentation were written with AI assistance. Saying so is
required, and it is also the honest description: the algorithms were specified
by the author, taken from the references in the proposal, and the AI wrote much
of the implementation from those specifications.

What that means for anyone reviewing this:

- Every algorithm is standard and named. Nothing here was invented by the AI,
  and nothing was ported from another codebase. There is no third-party code to
  trace, and no dependency beyond `moonbitlang/core`.
- The bugs listed above are the evidence that the output was not taken on
  trust. Six of them were found by tests rather than by reading, and two of
  those were only found because a test measured a *property* (bandwidth falls,
  the row sums vanish) rather than an expected value.
- The claims in the README are measurements, not estimates. The benchmark
  numbers, the iteration counts, the fill-in ratios and the memory figures all
  come from running the code, and the commands to reproduce them are in the
  README.

## Known limitations

- Reverse Cuthill-McKee is a heuristic and is not optimal. On the natural
  row-major numbering of a grid it is barely better, which is why
  `examples/reordering` reports the natural ordering as a baseline instead of
  only showing the favourable comparison.
- The iterative solvers have no restart strategy beyond GMRES's fixed restart,
  and no deflation. On a matrix with a few outlier eigenvalues they will be
  slow, and the caller has to notice and switch preconditioner.
- `CholeskyFactor::new` reads only the upper triangle of `A` and does not
  verify symmetry: a caller who passes an asymmetric matrix gets the factor of
  its symmetric part, silently. That is deliberate — a caller holding one
  triangle should not be forced to store both — and `new_checked` verifies
  symmetry first for a caller who is not sure. The sharp edge is that `new` is
  the shorter name.
- The reordering and PageRank routines pick their starting vertex by a linear
  scan for the minimum degree, which is `O(n)` per connected component. That is
  fine for the sizes in the examples and would need a bucket queue to scale.
- No 64-bit index type. `Int` is 32-bit, so a matrix with more than about two
  billion stored entries cannot be addressed. That is far beyond the intended
  range but it is a real ceiling.
