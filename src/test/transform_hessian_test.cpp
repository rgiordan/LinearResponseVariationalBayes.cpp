# include <Eigen/Dense>
# include <vector>

# include "transform_hessian.h"
# include "transform_hessian_test_functions.h"
# include "differentiate_jacobian.h"

# include <stan/math.hpp>
# include <stan/math/mix/mat/functor/hessian.hpp>

# include "gtest/gtest.h"

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using Eigen::Dynamic;
using Eigen::VectorXd;


// Tests for Eigen types.
#define EXPECT_VECTOR_EQ(x, y) \
  ASSERT_EQ(x.size(), y.size()); \
  for (int i = 0; i < x.size(); i ++) { \
    EXPECT_DOUBLE_EQ(x(i), y(i)) << " at index " << i; \
  }


#define EXPECT_MATRIX_EQ(x, y) \
  ASSERT_EQ(x.rows(), y.rows()); \
  ASSERT_EQ(x.cols(), y.cols()); \
  for (int i = 0; i < x.rows(); i ++) { \
    for (int j = 0; j < x.rows(); j ++) { \
      EXPECT_DOUBLE_EQ(x(i, j), y(i, j)) << " at index " << i << ", " << j; \
    } \
  }


TEST(y_to_x_to_y, is_inverse) {
  VectorXd y(2);
  y << 2, 3;
  VectorXd x = y_to_x(y);
  ASSERT_EQ(x.size(), y.size());
  VectorXd y_trans = x_to_y(x);
  ASSERT_EQ(y_trans.size(), x.size());
  EXPECT_VECTOR_EQ(y_trans, y);
  EXPECT_DOUBLE_EQ(f_of_y(y), f_of_x(x));
};


TEST(hessian_transforms, correct) {
  VectorXd y(2);
  y << 0.2, 0.3;

  VectorXd x(2);
  MatrixXd dxt_dy(2, 2);
  MatrixXd dyt_dx(2, 2);

  // Currently, stan::math::jacobian returns the transpose of the Jacobian!
  stan::math::set_zero_all_adjoints();
  stan::math::jacobian(y_to_x, y, x, dxt_dy);
  EXPECT_VECTOR_EQ(x, y_to_x(y));

  stan::math::jacobian(x_to_y, x, y, dyt_dx);
  EXPECT_VECTOR_EQ(y, x_to_y(x));

  double f_y_val;
  VectorXd df_dy(2);
  stan::math::gradient(f_of_y, y, f_y_val, df_dy);

  double f_x_val;
  VectorXd df_dx(2);
  stan::math::gradient(f_of_x, x, f_x_val, df_dx);

  EXPECT_DOUBLE_EQ(f_x_val, f_y_val);

  // Check the tranformation two different ways:

  // The inverse of dxt_dy is dyt_dx.
  VectorXd df_dx_from_jac2 = dyt_dx * df_dy;
  EXPECT_VECTOR_EQ(df_dx, df_dx_from_jac2);

  VectorXd df_dx_from_jac = dxt_dy.colPivHouseholderQr().solve(df_dy);
  EXPECT_VECTOR_EQ(df_dx, df_dx_from_jac);

  // Test the transformed hessian.

  vector<MatrixXd> d2x_dy2_vec = GetJacobianHessians(y_to_x, y);

  MatrixXd d2f_dy2(2, 2);
  VectorXd x_grad_unused(2);
  stan::math::hessian(f_of_y, y, f_y_val, df_dy, d2f_dy2);

  MatrixXd d2f_dx2 =
    transform_hessian(dxt_dy.transpose(), d2x_dy2_vec, df_dy, d2f_dy2);

  printf(".\n");
  MatrixXd d2f_dx2_test(2, 2);
  printf(".\n");
  stan::math::hessian(f_of_x, x, f_x_val, x_grad_unused, d2f_dx2_test);

  EXPECT_MATRIX_EQ(d2f_dx2, d2f_dx2_test);
};


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
