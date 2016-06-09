# ifndef EXPONENTIAL_FAMILIES_H
# define EXPONENTIAL_FAMILIES_H

#include <cmath>

#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include "variational_parameters.h"

#include <Eigen/Sparse>
typedef Eigen::Triplet<double> Triplet; // For populating sparse matrices

#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
// #include <stan/math/fwd/scal/fun/trigamma.hpp> // Missing!

using boost::math::lgamma;
using boost::math::digamma;
using boost::math::trigamma;

////////////////////////////////////////////
// Some helper functions

template <typename T> T multivariate_lgamma(T x, int p) {
  T result = log(M_PI) * p * (p - 1.0) / 4.0;
  for (int i = 1; i <= p; i++) {
    result += lgamma(x + 0.5 * (1 - (double)i));
  }
  return result;
}


template <typename T> T multivariate_digamma(T x, int p) {
  T result = 0.0;
  for (int i = 1; i <= p; i++) {
    result += digamma(x + 0.5 * (1 - (double)i));
  }
  return result;
}


// Note: this won't work with STAN hessians..
template <typename T> T multivariate_trigamma(T x, int p) {
 T result = 0.0;
  for (int i = 1; i <= p; i++) {
    result += trigamma(x + 0.5 * (1 - (double)i));
  }
  return result;
}


///////////////////////////
// Multivariate normals

// // Get Cov(mu) from its first and second moments.
// MatrixXd GetNormalCovariance(VectorXd const &e_mu, MatrixXd const &e_mu2);
//
// // Get Cov(mu_i1 mu_i2, mu_c mu_d) from the moment parameters of a multivariate
// // normal distribution.
// //
// // e_mu = E(mu)
// // cov_mu = E(mu mu^T) - E(mu) E(mu^T)
// double GetNormalFourthOrderCovariance(
//     VectorXd const &e_mu, MatrixXd const &cov_mu,
// 		int i1, int i2, int j1, int j2);
//
// // Get Cov(mu_i, mu_j1 mu_j2) from the moment parameters of a multivariate
// // normal distribution.
// //
// // e_mu = E(mu)
// // cov_mu = E(mu mu^T) - E(mu) E(mu^T)
// double GetNormalThirdOrderCovariance(
//     VectorXd const &e_mu, MatrixXd const &cov_mu, int i, int j1, int j2);


///////////////////////////////////////////
// Wishart distributions

// // Construct the covariance of the elements of a Wishart-distributed
// // matrix.
// //
// // Args:
// //   - v_par: The wishart matrix parameter.
// //   - n_par: The n parameter of the Wishart distribution.
// //
// // Returns:
// //   - Cov(w_i1_j1, w_i2_j2), where w_i1_j1 and w_i2_j2 are terms of the
// //     Wishart matrix parameterized by  and n_par.
// double GetWishartLinearCovariance(
//     MatrixXd const &v_par, double n_par, int i1, int j1, int i2, int j2);


// // Construct the covariance between the elements of a Wishart-distributed
// // matrix and the log determinant.  A little silly as a function, so
// // consider this documentation instead.
// //
// // Args:
// //   - v_par: A linearized representation of the upper triangular portion
// //            of the wishart parameter.
// //
// // Returns:
// //   - Cov(w_i1_i2, log(det(w)))
// double GetWishartLinearLogDetCovariance(MatrixXd const &v_par, int i1, int i2);
//
//
// // As above, but
// // Cov(log(det(w), log(det(w))))
// // ... where k is the dimension of the matrix.
// double GetWishartLogDetVariance(double n_par, int k);


template <typename T>
T GetELogDetWishart(MatrixXT<T> v_par, T n_par) {
  int k = v_par.rows();
  if (v_par.rows() != v_par.cols()) {
    throw std::runtime_error("V is not square");
  }
  T v_par_det = v_par.determinant();
  T e_log_det_lambda =
    log(v_par_det) + k * log(2) + multivariate_digamma(0.5 * n_par, k);
  return e_log_det_lambda;
};


template <typename T>
T GetWishartEntropy(MatrixXT<T> const &v_par, T const n_par) {
  int k = v_par.rows();
  if (v_par.rows() != v_par.cols()) {
    throw std::runtime_error("V is not square");
  }
  T det_v = v_par.determinant();
  T entropy = 0.5 * (k + 1) * log(det_v) +
              0.5 * k * (k + 1) * log(2) +
              multivariate_lgamma(0.5 * n_par, k) -
              0.5 * (n_par - k - 1) * multivariate_digamma(0.5 * n_par, k) +
              0.5 * n_par * k;
  return entropy;
}


////////////////////////////////////////
// Gamma distribution


template <typename T> T get_e_log_gamma(T alpha, T beta) {
  return digamma(alpha) - log(beta);
}


// Return a matrix with Cov((g, log(g))) where
// g ~ Gamma(alpha, beta) (parameterization E[g] = alpha / beta)
MatrixXd get_gamma_covariance(double alpha, double beta);


////////////////////////////
// Categorical

// Args:
//   p: A size k vector of the z probabilities.
// Returns:
//   The covariance matrix.
MatrixXd GetCategoricalCovariance(VectorXd p);


std::vector<Triplet> GetCategoricalCovarianceTerms(VectorXd p, int offset);


/////////////////////////////////
// Dirichlet

template <typename T> VectorXT<T> GetELogDirichlet(VectorXT<T> alpha) {
  // Args:
  //   - alpha: The dirichlet parameters Q ~ Dirichlet(alpha).  Should
  //     be a k-dimensional vector.
  // Returns:
  //   - A k-dimensional matrix representing E(log Q).

  int k = alpha.size();
  T alpha_sum = 0.0;
  for (int index = 0; index < k; index++) {
    alpha_sum += alpha(index);
  }
  T digamma_alpha_sum = boost::math::digamma(alpha_sum);
  VectorXT<T> e_log_alpha(k);
  for (int index = 0; index < k; index++) {
    e_log_alpha(index) = boost::math::digamma(alpha(index)) - digamma_alpha_sum;
  }
  return e_log_alpha;
}


template <typename T> T GetDirichletEntropy(VectorXT<T> alpha) {
  // Args:
  //   - alpha: The dirichlet parameters Q ~ Dirichlet(alpha).  Should
  //     be a k-dimensional vector.
  // Returns:
  //   - The entropy of the Dirichlet distribution.

  int k = alpha.size();
  T alpha_sum = alpha.sum();
  T entropy = 0.0;
  entropy += -lgamma(alpha_sum) - (k - alpha_sum) * digamma(alpha_sum);

  for (int index = 0; index < k; index++) {
    entropy += lgamma(alpha(index)) - (alpha(index) - 1) * digamma(alpha(index));
  }

  return entropy;
}

MatrixXd GetLogDirichletCovariance(VectorXd alpha);



///////////////////////////////////
// Coordinates and covariances for sparse matrices

// Assumes that e_mu and e_mu2_offset are stored linearly starting
// at their respective offsets.
std::vector<Triplet> get_mvn_covariance_terms(
    VectorXd e_mu, MatrixXd e_mu2, int e_mu_offset, int e_mu2_offset);

std::vector<Triplet> get_wishart_covariance_terms(
    MatrixXd v_par, double n_par, int e_lambda_offset, int e_log_det_lambda_offset);


std::vector<Triplet> get_gamma_covariance_terms(
    double alpha, double beta, int e_tau_offset, int e_log_tau_offset);

std::vector<Triplet> get_dirichlet_covariance_terms(VectorXd alpha, int offset);



# endif
