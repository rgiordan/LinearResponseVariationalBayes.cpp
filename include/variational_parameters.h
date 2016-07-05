# ifndef VARIATIONAL_PARAMETERS_H
# define VARIATIONAL_PARAMETERS_H

// Set to 0 to not instantiate.
// See https://github.com/stan-dev/math/issues/311#
# define INSTANTIATE_VARIATIONAL_PARAMETERS_H 1

# include <Eigen/Dense>
# include <vector>

# include "exponential_families.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Dynamic;

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;


# if INSTANTIATE_VARIATIONAL_PARAMETERS_H
  // For instantiation:
  # include <stan/math.hpp>
  # include "stan/math/fwd/scal.hpp"

  using var = stan::math::var;
  using fvar = stan::math::fvar<var>;
# endif

////////////////////////////////////
// Positive definite matrices

template <class T> class PosDefMatrixParameter {
// private:

public:
  int size;
  int size_ud;
  MatrixXT<T> mat;

  PosDefMatrixParameter(int size_): size(size_) {
    size_ud = size * (size + 1) / 2;
    mat = MatrixXT<T>::Zero(size, size);
  }

  PosDefMatrixParameter() {
    PosDefMatrixParameter(1);
  }

  template <typename TNumeric>
  void set(MatrixXT<TNumeric> set_value) {
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        T this_value = T(set_value(row, col));
        mat(row, col) = this_value;
        mat(col, row) = this_value;
      }
    }
  }

  template <typename TNumeric>
  void set_vec(VectorXT<TNumeric> set_value) {
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        T this_value = T(set_value(get_ud_index(row, col)));
        mat(row, col) = this_value;
        mat(col, row) = this_value;
      }
    }
  }

  // TODO: don't use the get() method, just access the matrix directly.
  MatrixXT<T> get() const {
    return mat;
  }

  VectorXT<T> get_vec() const {
    VectorXT<T> vec_value(size_ud);
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        vec_value(get_ud_index(row, col)) = mat(row, col);
      }
    }
    return vec_value;
  }

  template <typename Tnew>
  operator PosDefMatrixParameter<Tnew>() const {
    PosDefMatrixParameter<Tnew> pdmp = PosDefMatrixParameter<Tnew>(size);
    pdmp.set(mat);
    return pdmp;
  }

  // Set from an unconstrained matrix parameterization based on the Cholesky
  // decomposition.
  void set_unconstrained_vec(MatrixXT<T> set_value, T min_diag = 0.0) {
    if (min_diag < 0) {
      throw std::runtime_error("min_diag must be non-negative");
    }
    MatrixXT<T> chol_mat(size, size);
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        T this_value = T(set_value(get_ud_index(row, col)));
        if (col == row) {
          chol_mat(row, col) = exp(this_value);
        } else {
          chol_mat(row, col) = this_value;
          chol_mat(col, row) = 0.0;
        }
      }
    }
    mat = chol_mat * chol_mat.transpose();
    if (min_diag > 0) {
      for (int k = 0; k < size; k++) {
        mat(k, k) += min_diag;
      }
    }
  }

  VectorXT<T> get_unconstrained_vec(T min_diag = 0.0)  const {
    if (min_diag < 0) {
      throw std::runtime_error("min_diag must be non-negative");
    }
    VectorXT<T> free_vec(size_ud);
    MatrixXT<T> mat_minus_diag(mat);
    if (min_diag > 0) {
      for (int k = 0; k < size; k++) {
        mat_minus_diag(k, k) -= min_diag;
        if (mat_minus_diag(k, k) <= 0) {
          throw std::runtime_error("Posdef diagonal entry out of bounds");
        }
      }
    }
    MatrixXT<T> chol_mat = mat_minus_diag.llt().matrixL();
    for (int row=0; row < size; row++) {
      for (int col=0; col <= row; col++) {
        if (col == row) {
          free_vec(get_ud_index(row, col)) = log(chol_mat(row, col));
        } else {
          free_vec(get_ud_index(row, col)) = chol_mat(row, col);
        }
      }
    }
    return free_vec;
  }
};


/////////////////////////////////////
// Variational parameters.


//////// Gamma

template <class T> class GammaNatural {
public:

  T alpha;
  T beta;

  // For unconstrained encoding.
  T alpha_min;
  T beta_min;

  int encoded_size = 2;

  GammaNatural() {
    alpha = 3;
    beta = 3;

    alpha_min = 0;
    beta_min = 0;
  };

  template <typename Tnew> operator GammaNatural<Tnew>() const {
    GammaNatural<Tnew> gamma_new;
    gamma_new.alpha = alpha;
    gamma_new.beta = beta;
    gamma_new.alpha_min = alpha_min;
    gamma_new.beta_min = beta_min;

    return gamma_new;
  };

  VectorXT<T> encode_vector(bool unconstrained) const {
    VectorXT<T> vec(encoded_size);
    if (unconstrained) {
      vec(0) = log(alpha - alpha_min);
      vec(1) = log(beta - beta_min);
    } else {
      vec(0) = alpha;
      vec(1) = beta;
    }
    return vec;
  }

  template <typename Tnew>
  void decode_vector(VectorXT<Tnew> vec, bool unconstrained) {
    if (unconstrained) {
      alpha = exp(vec(0)) + alpha_min;
      beta = exp(vec(1)) + beta_min;
    } else {
      alpha = vec(0);
      beta = vec(1);
    }
  }
};


template <class T> class GammaMoments {
public:
  T e;
  T e_log;

  // For unconstrained encoding.
  T e_min;
  int encoded_size = 2;

  GammaMoments() {
    e = 1;
    e_log = 0;
    e_min = 0;
  };

  template <typename Tnew> operator GammaMoments<Tnew>() const {
    GammaMoments<Tnew> gamma_new;
    gamma_new.e = e;
    gamma_new.e_log = e_log;
    gamma_new.e_min = e_min;
    return gamma_new;
  };

  GammaMoments(GammaNatural<T> gamma_nat) {
    e = gamma_nat.alpha / gamma_nat.beta;
    e_log = get_e_log_gamma(gamma_nat.alpha, gamma_nat.beta);
  };

  T ExpectedLogLikelihood(T alpha, T beta) {
    return (alpha - 1) * e - beta * e_log;
  }

  VectorXT<T> encode_vector(bool unconstrained) const {
    VectorXT<T> vec(encoded_size);
    vec(1) = e_log;
    if (unconstrained) {
      vec(0) = log(e - e_min);
    } else {
      vec(0) = e;
    }
    return vec;
  }

  template <typename Tnew>
  void decode_vector(VectorXT<Tnew> vec, bool unconstrained) {
    e_log = vec(1);
    if (unconstrained) {
      e = exp(vec(0)) + e_min;
    } else {
      e = vec(0);
    }
  }

};


//////// Wishart

template <class T> class WishartNatural {
public:
  int dim;

  PosDefMatrixParameter<T> v;
  T n;

  // For unconstrained encoding.
  T n_min;
  T diag_min;
  int encoded_size;

  WishartNatural(int dim): dim(dim) {
    n = 0;
    v = PosDefMatrixParameter<T>(dim);
    v.mat = MatrixXT<T>::Zero(dim, dim);
    diag_min = 0;
    n_min = 0;
    encoded_size = 1 + v.size_ud;
  };

  WishartNatural() {
    WishartNatural(1);
  }

  template <typename Tnew> operator WishartNatural<Tnew>() const {
    WishartNatural<Tnew> wishart_new(dim);
    wishart_new.dim = dim;
    wishart_new.v = v;
    wishart_new.n = n;
    wishart_new.diag_min = diag_min;
    wishart_new.n_min = n_min;
    wishart_new.encoded_size = encoded_size;
    return wishart_new;
  };

  VectorXT<T> encode_vector(bool unconstrained) const {
    VectorXT<T> vec(encoded_size);
    if (unconstrained) {
      vec(0) = log(n - n_min);
      vec.segment(1, v.size_ud) = v.get_unconstrained_vec(diag_min);
    } else {
      vec(0) = n;
      vec.segment(1, v.size_ud) = v.get_vec();
    }
    return vec;
  };

  template <typename Tnew>
  void decode_vector(VectorXT<Tnew> vec, bool unconstrained) {
    VectorXT<T> v_vec = vec.segment(1, v.size_ud);
    if (unconstrained) {
      n = exp(vec(0)) + n_min;
      v.set_unconstrained_vec(v_vec, diag_min);
    } else {
      n = vec(0);
      v.set_vec(v_vec);
    }
  };
};


template <class T> class WishartMoments {
public:
  int dim;

  PosDefMatrixParameter<T> e;
  T e_log_det;

  // For the unconstrained parameterization.
  T diag_min;
  int encoded_size;

  WishartMoments(int dim): dim(dim) {
    e_log_det = 0;
    e = PosDefMatrixParameter<T>(dim);
    e.mat = MatrixXT<T>::Zero(dim, dim);
    diag_min = 0;
    encoded_size = 1 + e.size_ud;
  };

  WishartMoments() {
    WishartMoments(1);
  }

  WishartMoments(WishartNatural<T> wishart_nat) {
    dim = wishart_nat.dim;
    e = PosDefMatrixParameter<T>(dim);
    e.mat = wishart_nat.v.mat * wishart_nat.n;
    e_log_det = GetELogDetWishart(wishart_nat.v.mat, wishart_nat.n);
    encoded_size = 1 + e.size_ud;
  }

  template <typename Tnew> operator WishartMoments<Tnew>() const {
    WishartMoments<Tnew> wishart_new(dim);
    wishart_new.e = e;
    wishart_new.e_log_det = e_log_det;
    wishart_new.diag_min = diag_min;
    return wishart_new;
  };

  VectorXT<T> encode_vector(bool unconstrained) const {
    VectorXT<T> vec(encoded_size);
    vec(0) = e_log_det;
    if (unconstrained) {
      vec.segment(1, e.size_ud) = e.get_unconstrained_vec(diag_min);
    } else {
      vec.segment(1, e.size_ud) = e.get_vec();
    }
    return vec;
  };

  template <typename Tnew>
  void decode_vector(VectorXT<Tnew> vec, bool unconstrained) {
    VectorXT<T> e_vec = vec.segment(1, e.size_ud);
    e_log_det = vec(0);
    if (unconstrained) {
      e.set_unconstrained_vec(e_vec, diag_min);
    } else {
      e.set_vec(e_vec);
    }
  };
};


//////// Multivariate Normal


template <class T> class MultivariateNormalNatural {
public:
  int dim;
  VectorXT<T> loc;
  PosDefMatrixParameter<T> info;

  T diag_min;
  int encoded_size;

  MultivariateNormalNatural(int dim): dim(dim) {
    loc = VectorXT<T>::Zero(dim);
    info = PosDefMatrixParameter<T>(dim);
    info.mat = MatrixXT<T>::Zero(dim, dim);
    diag_min = 0;
    encoded_size = dim + info.size_ud;
  };

  MultivariateNormalNatural() {
    MultivariateNormalNatural(1);
  };

  // Convert to another type.
  template <typename Tnew> operator MultivariateNormalNatural<Tnew>() const {
    MultivariateNormalNatural<Tnew> mvn(dim);
    mvn.loc = loc.template cast <Tnew>();
    mvn.info.mat = info.mat.template cast<Tnew>();
    mvn.diag_min = diag_min;
    mvn.encoded_size = dim + info.size_ud;
    return mvn;
  };

  VectorXT<T> encode_vector(bool unconstrained) const {
    VectorXT<T> vec(encoded_size);
    vec.segment(0, dim) = loc;
    if (unconstrained) {
      vec.segment(dim, info.size_ud) = info.get_unconstrained_vec(diag_min);
    } else {
      vec.segment(dim, info.size_ud) = info.get_vec();
    }
    return vec;
  };

  template <typename Tnew>
  void decode_vector(VectorXT<Tnew> vec, bool unconstrained) {
    VectorXT<T> sub_vec = vec.segment(0, dim);
    loc = sub_vec;
    sub_vec = vec.segment(dim, info.size_ud);
    if (unconstrained) {
      info.set_unconstrained_vec(sub_vec, diag_min);
    } else {
      info.set_vec(sub_vec);
    }
  };
};


template <class T> class MultivariateNormalMoments {
public:
  int dim;
  VectorXT<T> e_vec;
  PosDefMatrixParameter<T> e_outer;

  T diag_min;
  int encoded_size;

  MultivariateNormalMoments(int dim): dim(dim) {
    e_vec = VectorXT<T>::Zero(dim);
    e_outer = PosDefMatrixParameter<T>(dim);
    e_outer.mat = MatrixXT<T>::Zero(dim, dim);
    diag_min = 0;
    encoded_size = dim + e_outer.size_ud;
  };

  MultivariateNormalMoments() {
    MultivariateNormalMoments(1);
  };

  // Set from a vector of another type.
  template <typename Tnew> MultivariateNormalMoments(VectorXT<Tnew> mean) {
    dim = mean.size();
    encoded_size = dim + e_outer.size_ud;
    e_vec = mean.template cast<T>();
    MatrixXT<T> e_vec_outer = e_vec * e_vec.transpose();
    e_outer = PosDefMatrixParameter<Tnew>(dim);
    e_outer.set(e_vec_outer);
    diag_min = 0;
  };

  // Set from natural parameters.
  MultivariateNormalMoments(MultivariateNormalNatural<T> mvn_nat) {
    dim = mvn_nat.dim;
    encoded_size = dim + e_outer.size_ud;
    e_vec = mvn_nat.loc;
    MatrixXT<T> e_outer_mat =
      e_vec * e_vec.transpose() + mvn_nat.info.mat.inverse();
    e_outer = PosDefMatrixParameter<T>(e_vec.size());
    e_outer.set(e_outer_mat);
    diag_min = 0;
  };

  // Convert to another type.
  template <typename Tnew> operator MultivariateNormalMoments<Tnew>() const {
    MultivariateNormalMoments<Tnew> mvn(dim);
    mvn.e_vec = e_vec.template cast <Tnew>();
    mvn.e_outer.mat = e_outer.mat.template cast<Tnew>();
    mvn.diag_min = diag_min;
    return mvn;
  };

  VectorXT<T> encode_vector(bool unconstrained) const {
    VectorXT<T> vec(encoded_size);
    vec.segment(0, dim) = e_vec;
    if (unconstrained) {
      vec.segment(dim, e_outer.size_ud) = e_outer.get_unconstrained_vec(diag_min);
    } else {
      vec.segment(dim, e_outer.size_ud) = e_outer.get_vec();
    }
    return vec;
  };

  template <typename Tnew>
  void decode_vector(VectorXT<Tnew> vec, bool unconstrained) {
    VectorXT<T> sub_vec = vec.segment(0, dim);
    e_vec = sub_vec;
    sub_vec = vec.segment(dim, e_outer.size_ud);
    if (unconstrained) {
      e_outer.set_unconstrained_vec(sub_vec, diag_min);
    } else {
      e_outer.set_vec(sub_vec);
    }
  };

  // If this MVN is distributed N(mean, info^-1), get the expected log likelihood.
  T ExpectedLogLikelihood(MultivariateNormalMoments<T> mean, WishartMoments<T> info) const {
    MatrixXT<T> mean_outer_prods = mean.e_vec * e_vec.transpose() +
                                   e_vec * mean.e_vec.transpose();
    return
      -0.5 * (info.e.mat * (e_outer.mat - mean_outer_prods + mean.e_outer.mat)).trace() +
      0.5 * info.e_log_det;
  };

  T ExpectedLogLikelihood(VectorXT<T> mean, WishartMoments<T> info) const {
    MatrixXT<T> mean_outer_prods = mean * e_vec.transpose() +
                                   e_vec * mean.transpose();
    MatrixXT<T> mean_outer = mean * mean.transpose();
    return
      -0.5 * (info.e.mat * (e_outer.mat - mean_outer_prods + mean_outer)).trace() +
      0.5 * info.e_log_det;
  };

  T ExpectedLogLikelihood(VectorXT<T> mean, MatrixXT<T> info) const {
    MatrixXT<T> mean_outer_prods = mean * e_vec.transpose() +
                                   e_vec * mean.transpose();
    MatrixXT<T> mean_outer = mean * mean.transpose();
    return
      -0.5 * (info * (e_outer.mat - mean_outer_prods + mean_outer)).trace() +
      0.5 * log(info.determinant());
  };
};


//////// Univariate Normal

template <class T> class UnivariateNormalNatural {
public:
  T loc;  // The expectation.
  T info; // The expectation of the square.

  T info_min;
  int encoded_size = 2;

  UnivariateNormalNatural() {
    loc = 0;
    info = 0;
    info_min = 0;
  };

  // Convert to another type.
  template <typename Tnew> operator UnivariateNormalNatural<Tnew>() const {
    UnivariateNormalNatural<Tnew> uvn;
    uvn.loc = loc;
    uvn.info = info;
    uvn.info_min = info_min;
    return uvn;
  };

  VectorXT<T> encode_vector(bool unconstrained) const {
    VectorXT<T> vec(encoded_size);
    if (unconstrained) {
      vec(0) = loc;
      vec(1) = log(info - info_min);
    } else {
      vec(0) = loc;
      vec(1) = info;
    }
    return vec;
  }

  template <typename Tnew>
  void decode_vector(VectorXT<Tnew> vec, bool unconstrained) {
    if (unconstrained) {
      loc = vec(0);
      info = exp(vec(1)) + info_min;
    } else {
      loc = vec(0);
      info = vec(1);
    }
  }
};


template <class T> class UnivariateNormalMoments {
public:
  T e;  // The expectation.
  T e2; // The expectation of the square.

  T e2_min;
  int encoded_size = 2;

  UnivariateNormalMoments() {
    e = 0;
    e2 = 0;
    e2_min = 0;
  };

  // Set from natural parameters.
  UnivariateNormalMoments(UnivariateNormalNatural<T> uvn_nat) {
    e = uvn_nat.loc;
    e2 = 1 / uvn_nat.info + pow(uvn_nat.loc, 2);
    e2_min = 0;
  };

  // Convert to another type.
  template <typename Tnew> operator UnivariateNormalMoments<Tnew>() const {
    UnivariateNormalMoments<Tnew> uvn;
    uvn.e = e;
    uvn.e2 = e2;
    uvn.e2_min = e2_min;
    return uvn;
  };

  VectorXT<T> encode_vector(bool unconstrained) const {
    VectorXT<T> vec(encoded_size);
    if (unconstrained) {
      vec(0) = e;
      vec(1) = log(e2 - e2_min);
    } else {
      vec(0) = e;
      vec(1) = e2;
    }
    return vec;
  }

  template <typename Tnew>
  void decode_vector(VectorXT<Tnew> vec, bool unconstrained) {
    if (unconstrained) {
      e = vec(0);
      e2 = exp(vec(1)) + e2_min;
    } else {
      e = vec(0);
      e2 = vec(1);
    }
  }


  // If this MVN is distributed N(mean, info^-1), get the expected log likelihood.
  T ExpectedLogLikelihood(UnivariateNormalMoments<T> mean, GammaMoments<T> info) const {
    return -0.5 * info.e * (e2 - 2 * mean.e * e + mean.e2) + 0.5 * info.e_log;
  };

  T ExpectedLogLikelihood(T mean, GammaMoments<T> info) const {
    return -0.5 * info.e * (e2 - 2 * mean * e + mean * mean) + 0.5 * info.e_log;
  };

  T ExpectedLogLikelihood(T mean, T info) const {
    return -0.5 * info * (e2 - 2 * mean * e + mean * mean) + 0.5 * log(info);
  };
};


# if INSTANTIATE_VARIATIONAL_PARAMETERS_H
  extern template class PosDefMatrixParameter<double>;
  extern template class PosDefMatrixParameter<var>;
  extern template class PosDefMatrixParameter<fvar>;

  extern template class GammaNatural<double>;
  extern template class GammaNatural<var>;
  extern template class GammaNatural<fvar>;

  extern template class GammaMoments<double>;
  extern template class GammaMoments<var>;
  extern template class GammaMoments<fvar>;

  extern template class WishartNatural<double>;
  extern template class WishartNatural<var>;
  extern template class WishartNatural<fvar>;

  extern template class WishartMoments<double>;
  extern template class WishartMoments<var>;
  extern template class WishartMoments<fvar>;

  extern template class MultivariateNormalNatural<double>;
  extern template class MultivariateNormalNatural<var>;
  extern template class MultivariateNormalNatural<fvar>;

  extern template class MultivariateNormalMoments<double>;
  extern template class MultivariateNormalMoments<var>;
  extern template class MultivariateNormalMoments<fvar>;

  extern template class UnivariateNormalNatural<double>;
  extern template class UnivariateNormalNatural<var>;
  extern template class UnivariateNormalNatural<fvar>;

  extern template class UnivariateNormalMoments<double>;
  extern template class UnivariateNormalMoments<var>;
  extern template class UnivariateNormalMoments<fvar>;
# endif



# endif
