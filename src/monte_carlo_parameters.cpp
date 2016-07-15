/////////////////////////////////
// Monte Carlo normal parameters

#include <monte_carlo_parameters.h>

# if INSTANTIATE_MONTE_CARLO_PARAMETERS_H
template VectorXT<double> MonteCarloNormalParameter::Evaluate(double, double) const;
template VectorXT<var> MonteCarloNormalParameter::Evaluate(var, var) const;
template VectorXT<fvar> MonteCarloNormalParameter::Evaluate(fvar, fvar) const;
# endif
