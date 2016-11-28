# include "monte_carlo_parameters.h"
# include "gtest/gtest.h"

# include <Eigen/Dense>
# include "test_eigen.h"

using Eigen::VectorXd;

struct MeanAndVar {
  double mean;
  double var;
};


MeanAndVar GetMeanAndVar(VectorXd vec) {
    MeanAndVar result; // (mean, variance)
    result.mean = vec.sum() / vec.size();
    double sum_sq = 0.0;
    for (int i = 0; i < vec.size(); i++) {
        sum_sq += pow(vec(i), 2);
    }
    // vec.unaryExpr([](double x) { return pow(x, 2); });  // Why u no work?
    result.var = sum_sq / vec.size() - pow(result.mean, 2);
    return result;
};


TEST(monte_carlo_parameters, is_correct) {
  int n_sim = 1000;
  MonteCarloNormalParameter norm_param(n_sim);

  MeanAndVar mean_and_var;
  mean_and_var = GetMeanAndVar(norm_param.std_draws);
  EXPECT_TRUE(abs(mean_and_var.mean) < 3 / sqrt(n_sim)) << mean_and_var.mean;
  EXPECT_TRUE(abs(mean_and_var.var - 1.0) < 6 / sqrt(n_sim))  << mean_and_var.var;

  float target_mean = 3.0;
  float target_var = 7.5;
  VectorXT<float> check_vec = norm_param.Evaluate(target_mean, target_var);

  VectorXd check_vec_double = check_vec.template cast<double>();
  mean_and_var = GetMeanAndVar(check_vec_double);
  EXPECT_TRUE(abs(mean_and_var.mean - target_mean) < 3 / sqrt(n_sim)) <<
    mean_and_var.mean;
  EXPECT_TRUE(abs(mean_and_var.var - target_var) < 6 / sqrt(n_sim))  <<
    mean_and_var.var;

  // Set a certain number of new draws.
  norm_param.SetDraws(10);
  EXPECT_TRUE(norm_param.std_draws.size() == 10) << "Set n_sim from number";
  // Set from a vector instead of a number.
  int new_n_sim = 500;
  VectorXd new_draws = VectorXd::Random(new_n_sim);
  norm_param.SetDraws(new_draws);
  EXPECT_VECTOR_EQ(new_draws, norm_param.std_draws, "Set from vector");

};


TEST(mvn_draw_from_std_normal_draw, is_correct) {
    int k = 2;
    VectorXd loc(k);
    loc << 1, 2;
    MatrixXd info(k, k);
    info  << 1, 0.2, 0.2, 1;
    MatrixXd cov = info.inverse();
    MatrixXd cov_chol_t = cov.llt().matrixL();
    MultivariateNormalNatural<double> mvn_nat(loc, info);

    MatrixXd id_cov = MatrixXd::Zero(k, k);
    for (int ind = 0; ind < k; ind++) {
        id_cov(ind, ind) = 1.0;
    }
    VectorXd id_mean = VectorXd::Zero(k);

    int n_sim = 3e5;
    boost::mt19937 rng;
    MatrixXd sample_cov = MatrixXd::Zero(k, k);
    VectorXd sample_mean = VectorXd::Zero(k);

    // Make sure both conversion methods do the same thing.
    VectorXd std_draw = stan::math::multi_normal_rng(id_mean, id_cov, rng);
    VectorXd mvn_draw_1 = MVNDrawFromStandardNormalDraw(std_draw, loc, cov_chol_t);
    VectorXd mvn_draw_2 = MVNDrawFromStandardNormalDraw(std_draw, mvn_nat);
    EXPECT_VECTOR_EQ(mvn_draw_1, mvn_draw_2, "Conversion from natural parameters");

    // Make sure the moments match.
    for (int n = 0; n  < n_sim; n++) {
        VectorXd std_draw = stan::math::multi_normal_rng(id_mean, id_cov, rng);
        VectorXd mvn_draw = MVNDrawFromStandardNormalDraw(std_draw, loc, cov_chol_t);
        sample_mean += mvn_draw;
        sample_cov += mvn_draw * mvn_draw.transpose();
    }
    sample_mean /= n_sim;
    sample_cov /= n_sim;
    sample_cov -= sample_mean * sample_mean.transpose();

    // TODO: there's a principled way to set these tolerances.
    EXPECT_VECTOR_NEAR(loc, sample_mean, 8 / sqrt(n_sim), "MVN mean");
    EXPECT_MATRIX_NEAR(cov, sample_cov, 25 / sqrt(n_sim), "MVN cov");
};


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
