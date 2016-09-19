# ifndef KAHAN_SUMMATION_H
# define KAHAN_SUMMATION_H

// Set to 0 to not instantiate.
// See https://github.com/stan-dev/math/issues/311#
# define INSTANTIATE_KAHAN_SUMMATION_H 1

# if INSTANTIATE_KAHAN_SUMMATION_H
  // For instantiation:
  # include "stan/math.hpp"
  # include "stan/math/fwd/scal.hpp"

  using var = stan::math::var;
  using fvar = stan::math::fvar<var>;
# endif


// Kahan summation
template <class T> class KahanAccumulator {
private:
  T correction;
  T intermediate_addend;
  T intermediate_value;

public:
  T value;

  KahanAccumulator() {
    correction = 0;
    intermediate_addend = 0;
    intermediate_value = 0;
    value = 0;
  }

  void Reset() {
    correction = 0;
    intermediate_addend = 0;
    intermediate_value = 0;
    value = 0;
  }

  T Add(T addend) {
    intermediate_addend = addend - correction;
    intermediate_value = value + intermediate_addend;
    correction = (intermediate_value - value) - intermediate_addend;
    value = intermediate_value;
  }
};

# if INSTANTIATE_KAHAN_SUMMATION_H
  extern template class KahanAccumulator<double>;
  extern template class KahanAccumulator<var>;
  extern template class KahanAccumulator<fvar>;
# endif

# endif
