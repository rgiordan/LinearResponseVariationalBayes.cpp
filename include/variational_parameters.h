# ifndef VARIATIONAL_PARAMETERS_H
# define VARIATIONAL_PARAMETERS_H

# include <Eigen/Dense>
# include <vector>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Dynamic;

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;


////////////////
// Scalar parameters

// TODO: get rid of the scalar and vector types which don't serve any purpose.

template <class T> class ScalarParameter {
private:
  T value;

public:
  std::string name;

  ScalarParameter(std::string name_): name(name_) {
    value = 0.0;
  };

  ScalarParameter() {
    ScalarParameter("Uninitialized");
  };

  template <typename TNumeric>
  void set(TNumeric new_value) {
    value = T(new_value);
  };

  T get() const {
    return value;
  };

  template <typename Tnew>
  operator ScalarParameter<Tnew>() const {
    ScalarParameter<Tnew> sp = ScalarParameter<Tnew>(name);
    sp.set(value);
    return sp;
  }
};


////////////////////////
// Vector parameters

template <class T> class VectorParameter {
private:
  VectorXT<T> value;

public:
  int size;
  std::string name;

  VectorParameter(int size_, std::string name_):
      size(size_), name(name_) {
    value = VectorXT<T>::Zero(size_);
  }

  VectorParameter(){
    VectorParameter(1, "uninitialized");
  }

  template <typename TNumeric>
  void set(VectorXT<TNumeric> new_value) {
    if (new_value.size() != size) {
      throw std::runtime_error("new_value must be the same size as the old");
    }
    for (int row=0; row < size; row++) {
      value(row) = T(new_value(row));
    }
  }

  template <typename TNumeric>
  void set_vec(VectorXT<TNumeric> new_value) {
    set(new_value);
  }

  VectorXT<T> get() const {
    return value;
  }

  VectorXT<T> get_vec() const {
    return value;
  }

  template <typename Tnew>
  operator VectorParameter<Tnew>() const {
    VectorParameter<Tnew> vp = VectorParameter<Tnew>(size, name);
    vp.set(value);
    return vp;
  }
};

////////////////////////////////////
// Positive definite matrices

// The index in a vector of lower diagonal terms of a particular matrix value.
int get_ud_index(int i, int j);

template <class T> class PosDefMatrixParameter {
// private:

public:
  int size;
  int size_ud;
  std::string name;
  MatrixXT<T> mat;

  PosDefMatrixParameter(int size_, std::string name_):
      size(size_), name(name_) {
    size_ud = size * (size + 1) / 2;
    mat = MatrixXT<T>::Zero(size, size);
  }

  PosDefMatrixParameter() {
    PosDefMatrixParameter(1, "uninitialized");
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

  // TODO: don't use the get() method.
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
    PosDefMatrixParameter<Tnew> pdmp = PosDefMatrixParameter<Tnew>(size, name);
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

template <class T> Gamma {
  int dim;

  // TODO: constrain these to always be consistent?
  T scale;
  T shape;

  T e;
  T e_log;

  Gamma() {
    e = 1;
    e_log = 0;
  }
}



template <class T> Wishart {
  int dim;

  // TODO: constrain these to always be consistent?
  PosDefMatrixParameter<T> v;
  T n;

  PosDefMatrixParameter<T> e_mat;
  T e_log_det;

  Wishart(int dim): dim(dim) {
    n = 0;
    v = MatrixXT<T>::Zero(dim, dim);

    e_log_det = 0;
    e_mat = MatrixXT<T>::Zero(dim, dim);
  }
}


template <class T> MultivatiateNormal {
  int dim;
  VectorXT<T> e_vec;
  PosDefMatrixParameter<T> e_outer;

  MultivariateNormal(int dim): dim(dim) {
    e_vec = VectorXT<T>::Zero(dim);
    e_outer = MatrixXT<T>::Zero(dim, dim);
  };

  MultivariateNormal() {
    MultivariateNormal(1);
  };

  // Set from a vector of another type.
  template <typename Tnew> MultivariateNormal(VectorXT<TNew> mean) const {
    dim = mean.size();
    e_vec = mean.template cast<T>();
    e_outer = PosDefMatrixParameter<T>(dim);
    e_outer.mat = e_vec * e_vec.transpose();
  };

  // Convert to another type.
  template <typename Tnew> operator MultivatiateNormal<Tnew>() const {
    MultivatiateNormal<Tnew> mvn(dim);
    mvn.e_vec = e_vec;
    mvn.e_outer = e_outer;
  };

  // If this MVN is distributed N(mean, info^-1), get the expected log likelihood.
  T ExpectedLogLikelihood(MultivatiateNormal<T> mean, Wishart<T> info) const {
    MatrixXT<T> mean_outer_prods = mean.e_vec * e_vec.transpose() +
                                   e_vec * mean.e_vec.transpose();
    return
      -0.5 * (info.e_mat * (e_outer.mat + mean_outer_prods + mean.e_outer.mat)).trace() +
      0.5 * info.e_log_det;
  };

  T ExpectedLogLikelihood(VectorXT<T> mean, Wishart<T> info) const {
    MatrixXT<T> mean_outer_prods = mean * e_vec.transpose() +
                                   e_vec * mean.transpose();
    MatrixXT<T> mean_outer = mean * mean.transpose();
    return
      -0.5 * (info.e_mat * (e_outer.mat + mean_outer_prods + mean_outer)).trace() +
      0.5 * info.e_log_det;
  };

  T ExpectedLogLikelihood(VectorXT<T> mean, MatrixXT<T> info) const {
    MatrixXT<T> mean_outer_prods = mean * e_vec.transpose() +
                                   e_vec * mean.transpose();
    MatrixXT<T> mean_outer = mean * mean.transpose();
    return
      -0.5 * (info * (e_outer.mat + mean_outer_prods + mean_outer)).trace() +
      0.5 * log(info.determinant());
  };
};


# endif
