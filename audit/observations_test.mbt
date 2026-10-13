///|
fn show_result(r : @solve.SolveResult) -> Unit {
  println(
    "x[0]=" +
    r.x[0].to_string() +
    ", residual=" +
    r.residual.to_string() +
    ", converged=" +
    r.converged.to_string() +
    ", iterations=" +
    r.iterations.to_string(),
  )
}

///|
test "r2 observations of solver failure modes" {
  let a = @ops.identity(1, 1.0)
  println("CG I large b:")
  let cg_large = Some(@solve.cg(a, [1.0e200])) catch {
    @err.SparseError::UnsupportedMatrix(msg) => {
      println("CG large b safely raises: " + msg)
      None
    }
    _ => None
  }
  if cg_large is Some(r) {
    show_result(r)
  }
  println("BiCGSTAB I large b:")
  let bicg_large = Some(@solve.bicgstab(a, [1.0e200])) catch {
    @err.SparseError::UnsupportedMatrix(msg) => {
      println("BiCGSTAB large b safely raises: " + msg)
      None
    }
    _ => None
  }
  if bicg_large is Some(r) {
    show_result(r)
  }
  let h = @ops.identity(1, 1.0e200)
  let g = @solve.gmres(h, [1.0])
  println("GMRES large A:")
  show_result(g)
  println(
    "GMRES true relative residual: " +
    @solve.relative_residual(h, g.x, [1.0]).to_string(),
  )
  let i2 = @ops.identity(2, 1.0)
  let b = [1.3e308, 1.3e308]
  println("CG RHS norm overflow:")
  let cg_overflow = Some(@solve.cg(i2, b, initial=Some([1.0e308, 1.0e308]))) catch {
    @err.SparseError::UnsupportedMatrix(msg) => {
      println("CG RHS norm overflow safely raises: " + msg)
      None
    }
    _ => None
  }
  if cg_overflow is Some(r) {
    show_result(r)
  }
  println("norm2 finite huge RHS: " + @solve.norm2(b).to_string())
  let small = Some(@solve.cg(a, [1.0e-200])) catch {
    @err.SparseError::UnsupportedMatrix(msg) => {
      println("CG tiny b raises: " + msg)
      None
    }
    _ => None
  }
  ignore(small)
  let adj = @graph.weighted_adjacency_from_edges(3, [
    (0, 1, 1.0e-200),
    (0, 2, 1.0e300),
  ])
  let nl = @graph.normalized_laplacian(adj)
  println(
    "normalized wide weights 01=" +
    nl.get_or(0, 1, 0.0).to_string() +
    ", 10=" +
    nl.get_or(1, 0, 0.0).to_string(),
  )
}
