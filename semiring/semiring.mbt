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
///   Floating-point semiring properties (such as associativity) hold within
///   standard IEEE-754 floating-point precision bounds.
/// - Negative infinity `-Infinity` satisfies `-Infinity * 1 = -Infinity`, while
///   `+Infinity * -Infinity = +Infinity` due to zero absorption.
pub(all) struct MinPlus {
  v : Double
  hi : Double
} derive(Debug, Eq)

///|
/// Wrap a distance. Use `MinPlus::infinity()` for "no path".
pub fn MinPlus::new(v : Double) -> MinPlus {
  if v.is_nan() {
    { v: 0.0 / 0.0, hi: 0.0 / 0.0, }
  } else if v.is_inf() {
    if v > 0.0 {
      MinPlus::infinity()
    } else {
      { v: -@double.infinity, hi: -@double.infinity, }
    }
  } else if Double::abs(v) < 1.0e300 {
    { v, hi: 0.0, }
  } else {
    let q = v / 1.0e300
    let hi = if q > 0.0 { Double::floor(q) } else { -Double::floor(-q) }
    let lo = v - hi * 1.0e300
    { v: lo, hi, }
  }
}

///|
/// The additive identity: no path, or an absent edge.
pub fn MinPlus::infinity() -> MinPlus {
  { v: @double.infinity, hi: @double.infinity, }
}

///|
/// The multiplicative identity: a path of length zero.
pub fn MinPlus::one() -> MinPlus {
  { v: 0.0, hi: 0.0, }
}

///|
/// True when this represents "no path".
pub fn MinPlus::is_infinite(self : MinPlus) -> Bool {
  (self.hi.is_inf() && self.hi > 0.0) || (self.v.is_inf() && self.v > 0.0)
}

///|
/// The unwrapped distance.
pub fn MinPlus::value(self : MinPlus) -> Double {
  if self.is_infinite() {
    @double.infinity
  } else if (self.hi.is_inf() && self.hi < 0.0) ||
    (self.v.is_inf() && self.v < 0.0) {
    -@double.infinity
  } else if self.hi == 0.0 {
    self.v
  } else {
    self.hi * 1.0e300 + self.v
  }
}

///|
pub impl Add for MinPlus with fn add(a : MinPlus, b : MinPlus) -> MinPlus {
  if a.is_infinite() && b.is_infinite() {
    MinPlus::infinity()
  } else if a.is_infinite() {
    b
  } else if b.is_infinite() {
    a
  } else if a.v.is_nan() || a.hi.is_nan() {
    a
  } else if b.v.is_nan() || b.hi.is_nan() {
    b
  } else if a.hi < b.hi {
    a
  } else if a.hi > b.hi {
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
  } else if a.v.is_nan() || b.v.is_nan() || a.hi.is_nan() || b.hi.is_nan() {
    { v: 0.0 / 0.0, hi: 0.0 / 0.0, }
  } else if (a.hi.is_inf() && a.hi < 0.0) ||
    (b.hi.is_inf() && b.hi < 0.0) ||
    (a.v.is_inf() && a.v < 0.0) ||
    (b.v.is_inf() && b.v < 0.0) {
    { v: -@double.infinity, hi: -@double.infinity, }
  } else {
    let raw_hi = a.hi + b.hi
    let raw_lo = a.v + b.v
    let carry = if Double::abs(raw_lo) >= 1.0e300 {
      let q = raw_lo / 1.0e300
      if q > 0.0 {
        Double::floor(q)
      } else {
        -Double::floor(-q)
      }
    } else {
      0.0
    }
    let new_hi = raw_hi + carry
    let new_lo = raw_lo - carry * 1.0e300
    if new_hi.is_inf() && new_hi > 0.0 {
      MinPlus::infinity()
    } else if new_hi.is_inf() && new_hi < 0.0 {
      { v: -@double.infinity, hi: -@double.infinity, }
    } else {
      { v: new_lo, hi: new_hi, }
    }
  }
}

///|
pub impl Default for MinPlus with fn default() -> MinPlus {
  MinPlus::infinity()
}
