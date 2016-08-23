# include "variational_parameters.h"
# include "gtest/gtest.h"

# include <string>
# include <Eigen/Dense>

# include "boost/random.hpp"
# include <stan/math.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

# include "test_eigen.h"

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

template <typename T> T Variance(VectorXT<T> vec) {
    T mean = vec.sum() / vec.size();
    T sum_sq = 0.0;
    for (int i = 0; i < vec.size(); i++) {
        sum_sq += pow(vec(i), 2);
    }
    // vec.unaryExpr([](double x) { return pow(x, 2); });  // Why u no work?
    T var = sum_sq / vec.size() - pow(mean, 2);
    return var;
};


void Sandbox() {
    GammaNatural<double> gamma_nat(2.1, 2.2);
    std::cout << "Hello:\n" << stan::math::gamma_log(3.0, 2.1, 2.2) << "\n";
    // Fails:
    std::cout << "Again:\n" << gamma_nat.log_lik(3.0) << "\n";
    double alpha = gamma_nat.alpha;
    double beta = gamma_nat.beta;
    std::cout << "Again:\n" << stan::math::gamma_log(3.0, alpha, beta) << "\n";
    std::cout << "False:\n" << stan::math::gamma_log<false>(3.0, 2.1, 2.2) << "\n";
    std::cout << "False:\n" << stan::math::gamma_log<false>(3.4, 2.1, 2.2) << "\n";

    // Looks like we shouldn't do this.
    std::cout << "True:\n" << stan::math::gamma_log<true>(3.0, 2.1, 2.2) << "\n";
    std::cout << "True:\n" << stan::math::gamma_log<true>(3.4, 2.1, 2.2) << "\n";

    VectorXd mean_vec(2);
    VectorXd obs(2);
    MatrixXd info_mat(2, 2);
    mean_vec << 1, 2;
    obs << 1.5, 2.5;
    info_mat << 2, 0.1, 0.1, 2;
    // std::cout << "MVN: " <<
    //     stan::math::multi_normal_prec_log(obs, mean_vec, info_mat) << "\n";
}


TEST(Gamma, log_lik) {
    GammaNatural<double> gamma_nat(6, 3);
    int num_i = 50;
    VectorXd log_lik_diff_vec(num_i);
    double obs_min = 1;
    double obs_max = 3;
    for (int i = 0; i < num_i; i++) {
        double obs = obs_min + i * (obs_max - obs_min) / num_i;
        GammaMoments<double> gamma_mom(obs, log(obs));
        log_lik_diff_vec(i) =
            gamma_nat.log_lik(obs) -
            gamma_mom.ExpectedLogLikelihood(gamma_nat.alpha, gamma_nat.beta);
    }
    EXPECT_NEAR(0, Variance(log_lik_diff_vec), 1e-12);
}


TEST(Wishart, log_lik) {
    MatrixXd v(2, 2);
    v << 2, 0.4, 0.4, 2;
    double n = 1.5;

    WishartNatural<double> wishart_nat(n, v);
    int num_i = 10;
    VectorXd log_lik_diff_vec(num_i * num_i);
    double diag_min = 0.5;
    double diag_max = 1.5;
    double offdiag_min = -0.1;
    double offdiag_max = 0.3;
    for (int i = 0; i < num_i; i++) {
        double obs_diag = diag_min + i * (diag_max - diag_min) / num_i;
        for (int j = 0; j < num_i; j++) {
            double obs_offdiag = offdiag_min + j * (offdiag_max - offdiag_min) / num_i;
            MatrixXd obs(2, 2);
            obs << obs_diag, obs_offdiag, obs_offdiag, obs_diag;
            double log_det_obs = log(obs.determinant());
            WishartMoments<double> wishart_mom(log_det_obs, obs);
            log_lik_diff_vec(j + i * num_i) =
                wishart_nat.log_lik(obs) -
                wishart_mom.ExpectedLogLikelihood(v, n);
        }
    }
    EXPECT_NEAR(0, Variance(log_lik_diff_vec), 1e-12);
}


TEST(MultivariateNormal, log_lik) {
    MatrixXd sigma(2, 2);
    sigma << 2, 0.4, 0.4, 2;

    VectorXd mean(2);
    mean << 0.2, 0.5;
    MatrixXd mean_outer = mean * mean.transpose();
    MatrixXd sigma_inv = sigma.inverse();
    MultivariateNormalNatural<double> mvn_nat(mean, sigma_inv);

    int num_i = 10;
    VectorXd log_lik_diff_vec(num_i * num_i);
    double obs_min = -0.1;
    double obs_max = 0.8;
    VectorXd obs(2);
    for (int i = 0; i < num_i; i++) {
        obs(0) = obs_min + i * (obs_max - obs_min) / num_i;
        for (int j = 0; j < num_i; j++) {
            obs(1) = obs_min + j * (obs_max - obs_min) / num_i;
            MatrixXd obs_outer = obs * obs.transpose();
            MultivariateNormalMoments<double> mvn_mom(obs, obs_outer);
            log_lik_diff_vec(j + i * num_i) =
                mvn_nat.log_lik(obs) -
                mvn_mom.ExpectedLogLikelihood(mean, sigma_inv);
        }
    }
    EXPECT_NEAR(0, Variance(log_lik_diff_vec), 1e-12);
}




int main(int argc, char **argv) {
    Sandbox();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
