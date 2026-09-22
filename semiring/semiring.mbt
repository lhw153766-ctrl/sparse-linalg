// Semirings, and why they are here.
//
// A sparse matrix product only ever needs two operations from its scalar type:
// something that plays the role of addition, and something that plays the role
// of multiplication. Nothing in the index arithmetic or the storage layout
// depends on those being numeric addition and multiplication.
//
// Choosing a different pair gives a different algorithm from the same code.
// Over the Boolean semiring, where "add" is OR and "multiply" is AND, the
// product of an adjacency matrix with itself answers reachability, and repeated
// squaring computes the transitive closure of a graph. Over the min-plus
// semiring, where "add" is minimum and "multiply" is addition, the same product
// computes shortest paths.
//
// These types exist so that the claim is testable rather than asserted: the
// storage and product code in this library is generic over the scalar type, and
// the tests in `semiring_test.mbt` run the same `spgemm` over all three.
//
// One implementation note that cost an afternoon: these are structs with named
// fields rather than tuple structs. A tuple struct does not pick up a core
// trait implementation in this toolchain, so `MinPlus(3.0) * MinPlus(5.0)`
// fails to resolve while `{ v: 3.0 } * { v: 5.0 }` works.

///|
/// The Boolean semiring: `add` is OR, `multiply` is AND, and the additive
/// identity is `false`.
///
/// Multiplying an adjacency matrix by itself over this semiring answers "is
/// there a walk of length exactly two", and summing successive powers answers
/// reachability.
pub(all) struct BoolSemiring {
  v : Bool
} derive(Debug, Eq)

///|
/// Wrap a `Bool`.
pub fn BoolSemiring::new(v : Bool) -> BoolSemiring {
  { v, }
}

///|
/// The unwrapped value.
pub fn BoolSemiring::value(self : BoolSemiring) -> Bool {
  self.v
}

///|
pub impl Add for BoolSemiring with fn add(a : BoolSemiring, b : BoolSemiring) -> BoolSemiring {
  { v: a.v || b.v, }
}

///|
pub impl Mul for BoolSemiring with fn mul(a : BoolSemiring, b : BoolSemiring) -> BoolSemiring {
  { v: a.v && b.v, }
}

///|
pub impl Default for BoolSemiring with fn default() -> BoolSemiring {
  { v: false, }
}

///|
/// The min-plus (tropical) semiring: `add` is minimum, `multiply` is addition,
/// and the additive identity is positive infinity.
///
/// This is the semiring that turns a matrix product into a shortest-path
/// computation. `Default` is infinity, so an absent entry correctly means "no
/// edge", and adding infinity to anything leaves it unchanged.
pub(all) struct MinPlus {
  v : Double
} derive(Debug, Eq)

///|
/// Wrap a distance. Use `MinPlus::infinity()` for "no path".
pub fn MinPlus::new(v : Double) -> MinPlus {
  { v, }
}

///|
/// The additive identity: no path, or an absent edge.
pub fn MinPlus::infinity() -> MinPlus {
  { v: 1.0e308, }
}

///|
/// The multiplicative identity: a path of length zero.
pub fn MinPlus::one() -> MinPlus {
  { v: 0.0, }
}

///|
/// True when this represents "no path".
pub fn MinPlus::is_infinite(self : MinPlus) -> Bool {
  self.v >= 1.0e308
}

///|
/// The unwrapped distance.
pub fn MinPlus::value(self : MinPlus) -> Double {
  self.v
}

///|
pub impl Add for MinPlus with fn add(a : MinPlus, b : MinPlus) -> MinPlus {
  { v: if a.v < b.v { a.v } else { b.v }, }
}

///|
pub impl Mul for MinPlus with fn mul(a : MinPlus, b : MinPlus) -> MinPlus {
  { v: a.v + b.v, }
}

///|
pub impl Default for MinPlus with fn default() -> MinPlus {
  { v: 1.0e308, }
}
