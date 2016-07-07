# ifndef MONTE_CARLO_PARAMETERS_H
# define MONTE_CARLO_PARAMETERS_H


// Set to 0 to not instantiate.
// See https://github.com/stan-dev/math/issues/311#
# define INSTANTIATE_MONTE_CARLO_PARAMETERS_H 1

/////////////////////////////////
// Monte Carlo normal parameters

#include <Eigen/Dense>

using Eigen::Dynamic;
template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <boost/random/normal_distribution.hpp>

# if INSTANTIATE_MONTE_CARLO_PARAMETERS_H
  // For instantiation:
  # include <stan/math.hpp>
  # include "stan/math/fwd/scal.hpp"

  using var = stan::math::var;
  using fvar = stan::math::fvar<var>;
# endif


// This is a typedef for a random number generator.
// Try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand
typedef boost::mt19937 RNGType;

template  <class T>
class MonteCarloNormalParameter {
public:
  MonteCarloNormalParameter(int n_sim) {
    SetDraws(n_sim);
  }

  void SetDraws(int n_sim) {
    std_draws = VectorXT<T>(n_sim);

    RNGType rng;
    boost::normal_distribution<> norm_dist(0,1);
    boost::variate_generator<RNGType, boost::normal_distribution<>>
      norm_rng(rng, norm_dist);
    for (int n=0; n < n_sim; n++) {
      std_draws(n) = norm_rng();
    }
  }

  VectorXT<T> Evaluate(T mean, T var) {
    VectorXT<T> output_vec(std_draws.size());
    for (int n=0; n < std_draws.size(); n++) {
      output_vec(n) = sqrt(var) * std_draws(n) + mean;
    }
    return output_vec;
  }

// private:
  VectorXT<T> std_draws;
};

# if INSTANTIATE_MONTE_CARLO_PARAMETERS_H
  extern template class MonteCarloNormalParameter<double>;
  extern template class MonteCarloNormalParameter<var>;
  extern template class MonteCarloNormalParameter<fvar>;
# endif


# endif
