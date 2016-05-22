# include <Eigen/Dense>
# include <vector>

# include <stan/math/fwd/scal/fun/pow.hpp>
# include <stan/math/fwd/scal/fun/exp.hpp>
# include <stan/math/fwd/scal/fun/log.hpp>
# include <stan/math/fwd/scal/fun/abs.hpp>

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using Eigen::Dynamic;
using Eigen::VectorXd;
template<typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template<typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

MatrixXd get_a_mat() {
  MatrixXd a(2, 2);
  a << 3, 1, 1, 2;
  return a;
}

const MatrixXd a_mat = get_a_mat();


struct y_to_x_functor {
  template <typename T> VectorXT<T> operator()(VectorXT<T> const &y) const {
    VectorXT<T> log_x = a_mat.template cast<T>() * y;
    VectorXT<T> x(log_x.size());
    for (int i = 0; i < x.size(); i++) {
      x(i) = exp(log_x(i));
    }
    return x;
  }
};
y_to_x_functor y_to_x;


// struct y_to_x_index_functor {
//   int i;
//
//   y_to_x_index_functor(int i): i(i) {};
//
//   template <typename T> T operator()(VectorXT<T> const &y) const {
//     return y_to_x(y)(i);
//   }
// };
// y_to_x_index_functor y_to_x_index(0);


struct x_to_y_functor {
  template <typename T> VectorXT<T> operator()(VectorXT<T> const &x) const {
    VectorXT<T> log_x(x.size());
    for (int i = 0; i < x.size(); i++) {
      log_x(i) = log(x(i));
    }
    VectorXT<T> y = a_mat.template cast<T>().ldlt().solve(log_x);
    return y;
  }
};
x_to_y_functor x_to_y;


struct f_of_y_functor {
  template <typename T> T operator()(VectorXT<T> const &y) const {
    return pow(y(0), 3) + pow(y(1), 2);
  }
};
f_of_y_functor f_of_y;


struct f_of_x_functor {
  template <typename T> T operator()(VectorXT<T> const &x) const {
    VectorXT<T> y = x_to_y(x);
    return f_of_y(y);
  }
};
f_of_x_functor f_of_x;
