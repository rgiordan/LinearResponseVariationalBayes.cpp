# include "variational_parameters.h"
# include "gtest/gtest.h"

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
  EXPECT_MATRIX_EQ(input, pd_mat.get());

  PosDefMatrixParameter<double> pd_mat2(k);
  VectorXd pd_mat_vec = pd_mat.get_vec();
  pd_mat2.set_vec(pd_mat_vec);
  EXPECT_MATRIX_EQ(pd_mat.get(), pd_mat2.get());

  pd_mat2.set(zero_mat);
  EXPECT_MATRIX_EQ(zero_mat, pd_mat2.get());
  pd_mat_vec = pd_mat.get_unconstrained_vec();
  pd_mat2.set_unconstrained_vec(pd_mat_vec);
  EXPECT_MATRIX_EQ(pd_mat.get(), pd_mat2.get());

  double min_diag = 2.0;
  pd_mat2.set(zero_mat);
  EXPECT_MATRIX_EQ(zero_mat, pd_mat2.get());
  pd_mat_vec = pd_mat.get_unconstrained_vec(min_diag);
  pd_mat2.set_unconstrained_vec(pd_mat_vec, min_diag);
  EXPECT_MATRIX_EQ(pd_mat.get(), pd_mat2.get());
}


TEST(MultivariateNormal, basic) {
  int dim = 3;
  VectorXd mean(dim);
  mean << 1, 2, 3;
  MatrixXd mean_outer = mean * mean.transpose();

  MultivariateNormal<double> mvn(dim);
  mvn.e_vec = mean;
  mvn.e_outer.set(mean_outer);

  MultivariateNormal<float> mvn_float(mean);
  EXPECT_VECTOR_EQ(mean, mvn_float.e_vec);
  EXPECT_MATRIX_EQ(mean_outer, mvn_float.e_outer.mat);

  MultivariateNormal<float> mvn_float2 = mvn;
  EXPECT_VECTOR_EQ(mvn.e_vec, mvn_float2.e_vec);
  EXPECT_MATRIX_EQ(mvn.e_outer.mat, mvn_float2.e_outer.mat);

  MatrixXd info_mat(3, 3);
  info_mat << 1,   0.1, 0.1,
              0.1, 1,   0.1,
              0.1, 0.1, 1;
  WishartMoments<double> info(dim);
  info.e.mat = info_mat;
  info.e_log_det = log(info_mat.determinant());

  VectorXd mean_center(dim);
  mean_center << 1.1, 2.1, 3.1;
  MultivariateNormal<double> mvn_center(mean_center);

  double log_lik_1 = mvn.ExpectedLogLikelihood(mvn_center, info);
  double log_lik_2 = mvn.ExpectedLogLikelihood(mean_center, info);
  double log_lik_3 = mvn.ExpectedLogLikelihood(mean_center, info_mat);

  EXPECT_DOUBLE_EQ(log_lik_1, log_lik_2);
  EXPECT_DOUBLE_EQ(log_lik_1, log_lik_3);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
