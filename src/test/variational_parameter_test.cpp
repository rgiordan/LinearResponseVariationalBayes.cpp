# include "variational_parameters.h"
# include "gtest/gtest.h"

# include <string>
# include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

# include "test_eigen.h"

TEST(PosDefMatrixParameter, basic) {
  int k = 3;
  MatrixXd zero_mat = MatrixXd::Zero(3, 3);
  PosDefMatrixParameter<double> pd_mat(k);
  MatrixXd input(k, k);
  input << 6, 1, 0,
           1, 7, 2,
           0, 2, 8;
  pd_mat.set(input);
  EXPECT_MATRIX_EQ(input, pd_mat.get(), "set input");

  PosDefMatrixParameter<double> pd_mat2(k);
  VectorXd pd_mat_vec = pd_mat.get_vec();
  pd_mat2.set_vec(pd_mat_vec);
  EXPECT_MATRIX_EQ(pd_mat.get(), pd_mat2.get(), "copy");

  pd_mat2.set(zero_mat);
  EXPECT_MATRIX_EQ(zero_mat, pd_mat2.get(), "set zero");
  pd_mat_vec = pd_mat.get_unconstrained_vec();
  pd_mat2.set_unconstrained_vec(pd_mat_vec);
  EXPECT_MATRIX_EQ(pd_mat.get(), pd_mat2.get(), "set unconstrained");

  double min_diag = 2.0;
  pd_mat2.set(zero_mat);
  EXPECT_MATRIX_EQ(zero_mat, pd_mat2.get(), "set zero");
  pd_mat_vec = pd_mat.get_unconstrained_vec(min_diag);
  pd_mat2.set_unconstrained_vec(pd_mat_vec, min_diag);
  EXPECT_MATRIX_EQ(pd_mat.get(), pd_mat2.get(), "set unconstrained");
}


TEST(MultivariateNormal, basic) {
  int dim = 3;
  VectorXd mean(dim);
  mean << 1, 2, 3;
  MatrixXd mean_outer = mean * mean.transpose();

  MultivariateNormalMoments<double> mvn(dim);
  mvn.e_vec = mean;
  mvn.e_outer.set(mean_outer);

  MultivariateNormalMoments<float> mvn_float(mean);
  EXPECT_FLOAT_VECTOR_EQ(mean, mvn_float.e_vec, "mean");
  EXPECT_FLOAT_MATRIX_EQ(mean_outer, mvn_float.e_outer.mat, "mean_outer");

  MultivariateNormalMoments<float> mvn_float2 = mvn;
  EXPECT_FLOAT_VECTOR_EQ(mvn.e_vec, mvn_float2.e_vec, "e_vec");
  EXPECT_FLOAT_MATRIX_EQ(mvn.e_outer.mat, mvn_float2.e_outer.mat, "e_outer");

  MatrixXd info_mat(dim, dim);
  info_mat << 1,   0.1, 0.1,
              0.1, 1,   0.1,
              0.1, 0.1, 1;
  WishartNatural<double> info_nat(dim);
  info_nat.v.mat = info_mat;
  info_nat.n = 1;
  WishartMoments<double> info(dim);
  info.e.mat = info_mat;
  info.e_log_det = log(info_mat.determinant());

  VectorXd mean_center(dim);
  mean_center << 1.1, 2.1, 3.1;
  MultivariateNormalMoments<double> mvn_center(mean_center);

  double log_lik_1 = mvn.ExpectedLogLikelihood(mvn_center, info);
  double log_lik_2 = mvn.ExpectedLogLikelihood(mean_center, info);
  double log_lik_3 = mvn.ExpectedLogLikelihood(mean_center, info_mat);

  EXPECT_DOUBLE_EQ(log_lik_1, log_lik_2);
  EXPECT_DOUBLE_EQ(log_lik_1, log_lik_3);

  // Test conversion from natural parameters.
  MultivariateNormalNatural<double> mvn_nat(dim);
  mvn_nat.loc = 2 * mean;
  mvn_nat.info.mat = info_mat;
  mvn = MultivariateNormalMoments<double>(mvn_nat);
  EXPECT_VECTOR_EQ(mvn_nat.loc, mvn.e_vec, "loc");
  MatrixXd expected_outer =
    info_mat.inverse() + mvn_nat.loc * mvn_nat.loc.transpose();
  EXPECT_MATRIX_EQ(expected_outer, mvn.e_outer.mat, "outer");
}


TEST(MultivariateNormalMoments, encoding) {
  int dim = 3;
  VectorXd vec(dim);
  vec << 1, 2, 3;
  MatrixXd mat(dim, dim);
  mat <<  1,   0.1, 0.1,
          0.1, 1,   0.1,
          0.1, 0.1, 1;

  MultivariateNormalMoments<double> mvn(dim);
  MultivariateNormalMoments<double> mvn_copy(dim);
  mvn.e_vec = vec;
  mvn.e_outer.set(mat);
  mvn_copy.diag_min = mvn.diag_min;

  for (int ind = 0; ind < 2; ind++) {
    bool unconstrained = (ind == 0 ? true: false);
    std::string unconstrained_str = (unconstrained ? "unconstrained": "constrained");
    VectorXd encoded_vec = mvn.encode_vector(unconstrained);
    mvn_copy.e_vec = VectorXd::Zero(dim);
    mvn_copy.e_outer.mat = MatrixXd::Zero(dim, dim);
    mvn_copy.decode_vector(encoded_vec, unconstrained);
    EXPECT_VECTOR_EQ(mvn.e_vec, mvn_copy.e_vec, unconstrained_str);
    EXPECT_MATRIX_EQ(mvn.e_outer.mat, mvn_copy.e_outer.mat, unconstrained_str);
  }
}


TEST(MultivariateNormalNatural, encoding) {
  int dim = 3;
  VectorXd vec(dim);
  vec << 1, 2, 3;
  MatrixXd mat(dim, dim);
  mat <<  1,   0.1, 0.1,
          0.1, 1,   0.1,
          0.1, 0.1, 1;

  MultivariateNormalNatural<double> mvn(dim);
  MultivariateNormalNatural<double> mvn_copy(dim);
  mvn.loc = vec;
  mvn.info.set(mat);
  mvn_copy.diag_min = mvn.diag_min;

  for (int ind = 0; ind < 2; ind++) {
    bool constrained = (ind == 0 ? true: false);
    std::string constrained_str = (constrained ? "constrained": "unconstrained");
    VectorXd encoded_vec = mvn.encode_vector(constrained);
    mvn_copy.loc = VectorXd::Zero(dim);
    mvn_copy.info.mat = MatrixXd::Zero(dim, dim);
    mvn_copy.decode_vector(encoded_vec, constrained);
    EXPECT_VECTOR_EQ(mvn.loc, mvn_copy.loc, constrained_str);
    EXPECT_MATRIX_EQ(mvn.info.mat, mvn_copy.info.mat, constrained_str);
  }
}


TEST(Wishart, basic) {
  MatrixXd v(3, 3);
  v << 1,   0.1, 0.1,
       0.1, 1,   0.1,
       0.1, 0.1, 1;
  double n = 3;
  WishartNatural<double> wishart_nat(3);
  wishart_nat.v.mat = v;
  wishart_nat.n = n;

  WishartMoments<double> wishart(wishart_nat);
  MatrixXd e_wishart = v * n;
  EXPECT_MATRIX_EQ(e_wishart, wishart.e.mat, "e_wishart");
  EXPECT_DOUBLE_EQ(GetELogDetWishart(v, n), wishart.e_log_det);

  // Test copying
  WishartNatural<float> wishart_nat_float(wishart_nat);
  EXPECT_FLOAT_MATRIX_EQ(wishart_nat_float.v.mat, wishart_nat.v.mat, "v");
  EXPECT_FLOAT_EQ(wishart_nat_float.n, wishart_nat.n);

  // Test copying
  WishartMoments<float> wishart_float(wishart);
  EXPECT_FLOAT_MATRIX_EQ(wishart_float.e.mat, wishart.e.mat, "e_mat");
  EXPECT_FLOAT_EQ(wishart_float.e_log_det, wishart.e_log_det);

}


TEST(Gamma, basic) {
  GammaMoments<double> gamma;
  gamma.e = 5;
  gamma.e_log = -3;
  GammaMoments<float> gamma2(gamma);
  EXPECT_FLOAT_EQ(gamma.e, gamma2.e);
  EXPECT_FLOAT_EQ(gamma.e_log, gamma2.e_log);

  GammaNatural<double> gamma_nat;
  double alpha = 4;
  double beta = 5;
  gamma_nat.alpha = alpha;
  gamma_nat.beta = beta;
  gamma = GammaMoments<double>(gamma_nat);
  EXPECT_DOUBLE_EQ(alpha / beta, gamma.e);
  EXPECT_DOUBLE_EQ(get_e_log_gamma(alpha, beta), gamma.e_log);

  // Just test that this runs.
  double e_log_lik = gamma.ExpectedLogLikelihood(alpha, beta);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
