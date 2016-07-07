# include "kahan_summation.h"

// Due to a bug in Stan, we can't include their headers here.
# if INSTANTIATE_KAHAN_SUMMATION_H
  template class KahanAccumulator<double>;
  template class KahanAccumulator<var>;
  template class KahanAccumulator<fvar>;
# endif
