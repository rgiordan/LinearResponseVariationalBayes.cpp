/////////////////////////////////
// Monte Carlo normal parameters

#include <monte_carlo_parameters.h>

# if INSTANTIATE_MONTE_CARLO_PARAMETERS_H
  template class MonteCarloNormalParameter<double>;
  template class MonteCarloNormalParameter<var>;
  template class MonteCarloNormalParameter<fvar>;
# endif
