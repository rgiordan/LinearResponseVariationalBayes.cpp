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
  // The matrix must still be positive definite after
  // subtracting this from the diagonal.
  mvn.diag_min = 0.2;
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
  // The matrix must still be positive definite after
  // subtracting this from the diagonal.
  mvn.diag_min = 0.2;
  mvn_copy.diag_min = mvn.diag_min;

  for (int ind = 0; ind < 2; ind++) {
    bool unconstrained = (ind == 0 ? true: false);
    std::string unconstrained_str = (unconstrained ? "unconstrained": "constrained");
    VectorXd encoded_vec = mvn.encode_vector(unconstrained);
    mvn_copy.loc = VectorXd::Zero(dim);
    mvn_copy.info.mat = MatrixXd::Zero(dim, dim);
    mvn_copy.decode_vector(encoded_vec, unconstrained);
    EXPECT_VECTOR_EQ(mvn.loc, mvn_copy.loc, unconstrained_str);
    EXPECT_MATRIX_EQ(mvn.info.mat, mvn_copy.info.mat, unconstrained_str);
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


TEST(WishartNatural, encoding) {
  int dim = 3;
  MatrixXd mat(dim, dim);
  mat <<  1,   0.1, 0.1,
          0.1, 1,   0.1,
          0.1, 0.1, 1;

  WishartNatural<double> wishart(dim);
  WishartNatural<double> wishart_copy(dim);
  wishart.n = 5.0;
  wishart.v.set(mat);
  wishart.diag_min = 0.2;
  wishart.n_min = 0.4;
  wishart_copy.diag_min = wishart.diag_min;
  wishart_copy.n_min = wishart.n_min;

  for (int ind = 0; ind < 2; ind++) {
    bool unconstrained = (ind == 0 ? true: false);
    std::string unconstrained_str = (unconstrained ? "unconstrained": "constrained");
    VectorXd encoded_vec = wishart.encode_vector(unconstrained);
    wishart_copy.n = 0.0;
    wishart_copy.v.mat = MatrixXd::Zero(dim, dim);
    wishart_copy.decode_vector(encoded_vec, unconstrained);
    EXPECT_DOUBLE_EQ(wishart.n, wishart_copy.n) << unconstrained_str;
    EXPECT_MATRIX_EQ(wishart.v.mat, wishart_copy.v.mat, unconstrained_str);
  }
}


TEST(WishartMoments, encoding) {
  int dim = 3;
  MatrixXd mat(dim, dim);
  mat <<  1,   0.1, 0.1,
          0.1, 1,   0.1,
          0.1, 0.1, 1;

  WishartMoments<double> wishart(dim);
  WishartMoments<double> wishart_copy(dim);
  wishart.e_log_det = 5.0;
  wishart.e.set(mat);
  wishart.diag_min = 0.2;
  wishart_copy.diag_min = wishart.diag_min;

  for (int ind = 0; ind < 2; ind++) {
    bool unconstrained = (ind == 0 ? true: false);
    std::string unconstrained_str = (unconstrained ? "unconstrained": "constrained");
    VectorXd encoded_vec = wishart.encode_vector(unconstrained);
    wishart_copy.e_log_det = 0.0;
    wishart_copy.e.mat = MatrixXd::Zero(dim, dim);
    wishart_copy.decode_vector(encoded_vec, unconstrained);
    EXPECT_DOUBLE_EQ(wishart.e_log_det, wishart_copy.e_log_det) << unconstrained_str;
    EXPECT_MATRIX_EQ(wishart.e.mat, wishart_copy.e.mat, unconstrained_str);
  }
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


TEST(GammaNatural, encoding) {
  GammaNatural<double> gamma;
  GammaNatural<double> gamma_copy;
  gamma.alpha = 3.0;
  gamma.beta = 4.0;
  gamma.alpha_min = 0.1;
  gamma.beta_min = 0.2;
  gamma_copy.alpha_min = gamma.alpha_min;
  gamma_copy.beta_min = gamma.beta_min;

  for (int ind = 0; ind < 2; ind++) {
    bool unconstrained = (ind == 0 ? true: false);
    std::string unconstrained_str = (unconstrained ? "unconstrained": "constrained");
    VectorXd encoded_vec = gamma.encode_vector(unconstrained);
    gamma_copy.alpha = 0.0;
    gamma_copy.beta = 0.0;
    gamma_copy.decode_vector(encoded_vec, unconstrained);
    EXPECT_DOUBLE_EQ(gamma.alpha, gamma_copy.alpha) << unconstrained_str;
    EXPECT_DOUBLE_EQ(gamma.beta, gamma_copy.beta) << unconstrained_str;
  }
}


TEST(GammaMoments, encoding) {
  GammaMoments<double> gamma;
  GammaMoments<double> gamma_copy;
  gamma.e = 3.0;
  gamma.e_log = 4.0;
  gamma.e_min = 0.1;
  gamma_copy.e_min = gamma.e_min;

  for (int ind = 0; ind < 2; ind++) {
    bool unconstrained = (ind == 0 ? true: false);
    std::string unconstrained_str = (unconstrained ? "unconstrained": "constrained");
    VectorXd encoded_vec = gamma.encode_vector(unconstrained);
    gamma_copy.e = 0.0;
    gamma_copy.e_log = 0.0;
    gamma_copy.decode_vector(encoded_vec, unconstrained);
    EXPECT_DOUBLE_EQ(gamma.e, gamma_copy.e) << unconstrained_str;
    EXPECT_DOUBLE_EQ(gamma.e_log, gamma_copy.e_log) << unconstrained_str;
  }
}


TEST(UnivariateNormal, basic) {
  UnivariateNormalNatural<double> uvn_nat;
  uvn_nat.loc = 3.0;
  uvn_nat.info = 4.0;

  UnivariateNormalMoments<double> uvn(uvn_nat);
  EXPECT_DOUBLE_EQ(uvn_nat.loc, uvn.e);
  EXPECT_DOUBLE_EQ(1 / uvn_nat.info, uvn.e2 - pow(uvn.e, 2));

  UnivariateNormalMoments<float> uvn_float(uvn);
  EXPECT_FLOAT_EQ(uvn.e, uvn_float.e);
  EXPECT_FLOAT_EQ(uvn.e2, uvn_float.e2);
}


TEST(UnivariateNormalMoments, encoding) {
  UnivariateNormalMoments<double> uvn;
  UnivariateNormalMoments<double> uvn_copy;
  uvn.e = 3.0;
  uvn.e2 = 4.0;
  uvn.e2_min = 0.1;
  uvn_copy.e2_min = uvn.e2_min;

  for (int ind = 0; ind < 2; ind++) {
    bool unconstrained = (ind == 0 ? true: false);
    std::string unconstrained_str = (unconstrained ? "unconstrained": "constrained");
    VectorXd encoded_vec = uvn.encode_vector(unconstrained);
    uvn_copy.e = 0.0;
    uvn_copy.e2 = 0.0;
    uvn_copy.decode_vector(encoded_vec, unconstrained);
    EXPECT_DOUBLE_EQ(uvn.e, uvn_copy.e) << unconstrained_str;
    EXPECT_DOUBLE_EQ(uvn.e2, uvn_copy.e2) << unconstrained_str;
  }
}


TEST(UnivariateNormalNatural, encoding) {
  UnivariateNormalNatural<double> uvn;
  UnivariateNormalNatural<double> uvn_copy;
  uvn.loc = 3.0;
  uvn.info = 4.0;
  uvn.info_min = 0.1;
  uvn_copy.info_min = uvn.info_min;

  for (int ind = 0; ind < 2; ind++) {
    bool unconstrained = (ind == 0 ? true: false);
    std::string unconstrained_str = (unconstrained ? "unconstrained": "constrained");
    VectorXd encoded_vec = uvn.encode_vector(unconstrained);
    uvn_copy.loc = 0.0;
    uvn_copy.info = 0.0;
    uvn_copy.decode_vector(encoded_vec, unconstrained);
    EXPECT_DOUBLE_EQ(uvn.loc, uvn_copy.loc) << unconstrained_str;
    EXPECT_DOUBLE_EQ(uvn.info, uvn_copy.info) << unconstrained_str;
  }
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
