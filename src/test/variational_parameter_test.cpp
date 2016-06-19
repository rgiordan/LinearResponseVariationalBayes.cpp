# include "variational_parameters.h"
# include "gtest/gtest.h"

# include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

# include "test_eigen.h"

TEST(PosDefMatrixParameter, basic) {
  int k = 3;
  MatrixXd zero_mat = MatrixXd::Zero(3, 3);
  PosDefMatrixParameter<double> pd_mat(k, "test");
  MatrixXd input(k, k);
  input << 6, 1, 0,
           1, 7, 2,
           0, 2, 8;
  pd_mat.set(input);
  EXPECT_MATRIX_EQ(input, pd_mat.get());

  PosDefMatrixParameter<double> pd_mat2(k, "test");
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


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
