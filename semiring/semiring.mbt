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
///
/// Semantics and IEEE-754 contract:
/// - Elements represent floating-point weights / path lengths.
/// - The additive identity is positive infinity (`MinPlus::infinity()`).
/// - The multiplicative identity is 0.0 (`MinPlus::one()`).
/// - Positive infinity is the unique absorbing zero element: `+Infinity * x = +Infinity` for all x.
/// - Ordinary negative and positive finite numbers preserve exact Double precision.
///   Floating-point arithmetic adheres to IEEE-754 semantics. IEEE-754 floating-point
///   addition does not strictly satisfy associativity or distributivity due to
///   rounding errors and extreme dynamic scale discrepancies. Within the representable
///   normal dynamic range, algebraic identities hold up to standard floating-point
///   precision; additions overflowing Double dynamic capacity saturate to +/-Infinity.
/// - Negative infinity `-Infinity` satisfies `-Infinity * 1 = -Infinity`, while
///   `+Infinity * -Infinity = +Infinity` due to zero absorption.
pub struct MinPlus {
  v : Double
} derive(Debug, Eq)

///|
/// Wrap a distance. Use `MinPlus::infinity()` for "no path".
pub fn MinPlus::new(v : Double) -> MinPlus {
  if v.is_nan() {
    { v: 0.0 / 0.0, }
  } else if v.is_inf() {
    if v > 0.0 {
      MinPlus::infinity()
    } else {
      { v: -@double.infinity, }
    }
  } else if v == 0.0 {
    // Canonicalize -0.0 to 0.0
    { v: 0.0, }
  } else {
    { v, }
  }
}

///|
/// The additive identity: no path, or an absent edge.
pub fn MinPlus::infinity() -> MinPlus {
  { v: @double.infinity, }
}

///|
/// The multiplicative identity: a path of length zero.
pub fn MinPlus::one() -> MinPlus {
  { v: 0.0, }
}

///|
/// True when this represents "no path".
pub fn MinPlus::is_infinite(self : MinPlus) -> Bool {
  self.v.is_inf() && self.v > 0.0
}

///|
/// The unwrapped distance.
pub fn MinPlus::value(self : MinPlus) -> Double {
  self.v
}

///|
pub impl Add for MinPlus with fn add(a : MinPlus, b : MinPlus) -> MinPlus {
  if a.is_infinite() {
    b
  } else if b.is_infinite() {
    a
  } else if a.v.is_nan() {
    a
  } else if b.v.is_nan() {
    b
  } else if a.v <= b.v {
    a
  } else {
    b
  }
}

///|
pub impl Mul for MinPlus with fn mul(a : MinPlus, b : MinPlus) -> MinPlus {
  if a.is_infinite() || b.is_infinite() {
    MinPlus::infinity()
  } else if a.v.is_nan() || b.v.is_nan() {
    { v: 0.0 / 0.0, }
  } else if a.v == 0.0 {
    b
  } else if b.v == 0.0 {
    a
  } else if (a.v.is_inf() && a.v < 0.0) || (b.v.is_inf() && b.v < 0.0) {
    { v: -@double.infinity, }
  } else {
    let sum = a.v + b.v
    if sum == 0.0 {
      { v: 0.0, }
    } else if sum.is_inf() && sum > 0.0 {
      MinPlus::infinity()
    } else if sum.is_inf() && sum < 0.0 {
      { v: -@double.infinity, }
    } else {
      { v: sum, }
    }
  }
}

///|
pub impl Default for MinPlus with fn default() -> MinPlus {
  MinPlus::infinity()
}
