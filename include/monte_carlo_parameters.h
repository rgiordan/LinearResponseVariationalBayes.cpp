# ifndef MONTE_CARLO_PARAMETERS_H
# define MONTE_CARLO_PARAMETERS_H

/////////////////////////////////
// Monte Carlo normal parameters

#include <Eigen/Dense>

using Eigen::Dynamic;
template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
// #include <boost/random/linear_congruential.hpp>
// #include <boost/random/variate_generator.hpp>
// #include <boost/generator_iterator.hpp>
#include <boost/random/normal_distribution.hpp>


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


# endif
