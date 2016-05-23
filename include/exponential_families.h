# ifndef EXPONENTIAL_FAMILIES_H
# define EXPONENTIAL_FAMILIES_H

#include <cmath>

#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include "variational_parameters.h"

#include <Eigen/Sparse>
typedef Eigen::Triplet<double> Triplet; // For populating sparse matrices


# include <stan/math.hpp>

#include <stan/math/fwd/scal/fun/lgamma.hpp>
#include <stan/math/fwd/scal/fun/digamma.hpp>
// #include <stan/math/fwd/scal/fun/trigamma.hpp> // Missing!

using boost::math::lgamma;
using boost::math::digamma;
using boost::math::trigamma;

using var = stan::math::var;
using fvar = stan::math::fvar<var>;

////////////////////////////////////////////
// Some helper functions

template <typename T> T multivariate_lgamma(T x, int p);
template <typename T> T multivariate_digamma(T x, int p);

// You can use this template to instantiate functions in static libraries.
// template double multivariate_lgamma(double, int);
// template var multivariate_lgamma(var, int);
// template fvar multivariate_lgamma(fvar, int);

// Note: this won't work with STAN hessians..
template <typename T> T multivariate_trigamma(T x, int p);


///////////////////////////
// Multivariate normals

MatrixXd GetNormalCovariance(VectorXd const &e_mu, MatrixXd const &e_mu2);


double GetNormalFourthOrderCovariance(
    VectorXd const &e_mu, MatrixXd const &cov_mu,
		int i1, int i2, int j1, int j2);

// Get Cov(mu_i, mu_j1 mu_j2) from the moment parameters of a multivariate
// normal distribution.
//
// e_mu = E(mu)
// cov_mu = E(mu mu^T) - E(mu) E(mu^T)
double GetNormalThirdOrderCovariance(
    VectorXd const &e_mu, MatrixXd const &cov_mu, int i, int j1, int j2);


///////////////////////////////////////////
// Wishart distributions

// Construct the covariance of the elements of a Wishart-distributed
// matrix.
//
// Args:
//   - v_par: The wishart matrix parameter.
//   - n_par: The n parameter of the Wishart distribution.
//
// Returns:
//   - Cov(w_i1_j1, w_i2_j2), where w_i1_j1 and w_i2_j2 are terms of the
//     Wishart matrix parameterized by  and n_par.
double GetWishartLinearCovariance(
    MatrixXd const &v_par, double n_par, int i1, int j1, int i2, int j2);


// Construct the covariance between the elements of a Wishart-distributed
// matrix and the log determinant.  A little silly as a function, so
// consider this documentation instead.
//
// Args:
//   - v_par: A linearized representation of the upper triangular portion
//            of the wishart parameter.
//
// Returns:
//   - Cov(w_i1_i2, log(det(w)))
double GetWishartLinearLogDetCovariance(MatrixXd const &v_par, int i1, int i2);


// As above, but
// Cov(log(det(w), log(det(w))))
// ... where k is the dimension of the matrix.
double GetWishartLogDetVariance(double n_par, int k);


template <typename T>
T GetELogDetWishart(MatrixXT<T> v_par, T n_par);

template <typename T>
T GetWishartEntropy(MatrixXT<T> const &v_par, T const n_par);

////////////////////////////////////////
// Gamma distribution

template <typename T> T get_e_log_gamma(T alpha, T beta);


// Return a matrix with Cov((g, log(g))) where
// g ~ Gamma(alpha, beta) (parameterization E[g] = alpha / beta)
MatrixXd get_gamma_covariance(double alpha, double beta);

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

# endif
