# ifndef VARIATIONAL_PARAMETERS_H
# define VARIATIONAL_PARAMETERS_H

// Set to 0 to not instantiate.
// See https://github.com/stan-dev/math/issues/311#
# define INSTANTIATE_VARIATIONAL_PARAMETERS_H 1

# include <Eigen/Dense>
# include <vector>
# include <iostream>
# include <string>
# include <limits>

# include "exponential_families.h"
# include "box_parameters.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Dynamic;

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

# define DOUBLE_INF std::numeric_limits<double>::infinity()

# if INSTANTIATE_VARIATIONAL_PARAMETERS_H
// For instantiation:
# include "stan/math/mix/mat.hpp"

using var = stan::math::var;
using fvar = stan::math::fvar<var>;
# endif


template <typename T> void PrintVector(VectorXT<T> vec, std::string name) {
    std::cout << name << ":\n(";
    for (int i=0; i < vec.size() - 1; i++) {
        std::cout << vec(i) << "\n";
    }
    std::cout << vec(vec.size() - 1)  << ")\n";
};


template <typename T> void PrintMatrix(MatrixXT<T> mat, std::string name) {
    std::cout << name << ":\n";
    for (int i=0; i < mat.rows(); i++) {
        std::cout << "[ ";
        for (int j=0; j < mat.cols() - 1; j++) {
            std::cout << mat(i, j) << ",\t";
        }
        std::cout << mat(i, mat.cols() - 1) << " ]\n";
    }
};


////////////////////////////////////
// Positive definite matrices

template <class T> class PosDefMatrixParameter {
private:
    void Init(int _size) {
        size = _size;
        size_ud = size * (size + 1) / 2;
        mat = MatrixXT<T>::Zero(size, size);
        scale_cholesky = false;
    }
public:
    int size;
    int size_ud;
    MatrixXT<T> mat;

    // An alternative unconstrained parameterization.
    bool scale_cholesky;

    PosDefMatrixParameter(int _size): size(_size) {
        Init(_size);
    }

    PosDefMatrixParameter() {
        Init(1);
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
    void set_cholesky_unconstrained_vec(VectorXT<T> set_value, T min_diag = 0.0) {
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

    VectorXT<T> get_cholesky_unconstrained_vec(T min_diag = 0.0)  const {
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
    };

    // Set from an unconstrained matrix parameterization based on a Cholesky
    // decomposition where all the diagonals are scaled to one.
    // This is also called an "LDL decomposition".
    void set_scale_cholesky_unconstrained_vec(VectorXT<T> set_value, T min_diag = 0.0) {
        if (min_diag < 0) {
            throw std::runtime_error("min_diag must be non-negative");
        }
        MatrixXT<T> chol_mat(size, size);
        MatrixXT<T> scale_mat = MatrixXT<T>::Zero(size, size);
        for (int row=0; row < size; row++) {
            for (int col=0; col <= row; col++) {
                T this_value = T(set_value(get_ud_index(row, col)));
                if (col == row) {
                    scale_mat(row, col) = exp(this_value);
                    chol_mat(row, col) = 1.0;
                } else {
                    scale_mat(row, col) = 0.0;
                    chol_mat(row, col) = this_value;
                    chol_mat(col, row) = 0.0;
                }
            }
        }
        MatrixXT<T> scaled_chol_mat = scale_mat * chol_mat;
        mat = scaled_chol_mat * scaled_chol_mat.transpose();
        if (min_diag > 0) {
            for (int k = 0; k < size; k++) {
                mat(k, k) += min_diag;
            }
        }
    }

    VectorXT<T> get_scale_cholesky_unconstrained_vec(T min_diag = 0.0)  const {
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

        MatrixXT<T> scale_vec = VectorXT<T>::Zero(size, size);
        MatrixXT<T> unscale_mat = MatrixXT<T>::Zero(size, size);
        for (int k = 0; k < size; k++) {
            scale_vec(k) = chol_mat(k, k);
            unscale_mat(k, k) = 1 / chol_mat(k, k);
        }

        MatrixXT<T> scale_free_chol_mat = unscale_mat * chol_mat;
        for (int row=0; row < size; row++) {
            for (int col=0; col <= row; col++) {
                if (col == row) {
                    free_vec(get_ud_index(row, col)) = log(scale_vec(row));
                } else {
                    free_vec(get_ud_index(row, col)) =
                        scale_free_chol_mat(row, col);
                }
            }
        }
        return free_vec;
    };

    void set_unconstrained_vec(VectorXT<T> set_value, T min_diag = 0.0) {
        if (!scale_cholesky) {
            set_cholesky_unconstrained_vec(set_value, min_diag);
        } else {
            set_scale_cholesky_unconstrained_vec(set_value, min_diag);
        }
    };

    VectorXT<T> get_unconstrained_vec(T min_diag = 0.0)  const {
        if (!scale_cholesky) {
            return get_cholesky_unconstrained_vec(min_diag);
        } else {
            return get_scale_cholesky_unconstrained_vec(min_diag);
        }
    }

};


/////////////////////////////////////
// Variational parameters.

template <class T> class Parameter {
public:
    int encoded_size;
    virtual VectorXT<T> encode_vector(bool unconstrained) const = 0;
    virtual void decode_vector(VectorXT<T> vec, bool unconstrained) = 0;

    Parameter(int encoded_size): encoded_size(encoded_size) {};
    Parameter(): encoded_size(0) {};
};


//////// Gamma

template <class T> class GammaNatural: public Parameter<T> {
    void Init() {
        alpha = 3;
        beta = 3;

        alpha_min = 0;
        beta_min = 0;
        alpha_max = DOUBLE_INF;
        beta_max = DOUBLE_INF;
        encoded_size = 2;
    }
public:

    using Parameter<T>::encoded_size;

    // Alpha and beta are shape and rate parameters, i.e.
    // E(obs) = alpha / beta
    T alpha;
    T beta;

    // For unconstrained encoding.
    double alpha_min;
    double beta_min;
    double alpha_max;
    double beta_max;

    GammaNatural() {
        Init();
    };

    GammaNatural(T _alpha, T _beta) {
        Init();
        set(_alpha, _beta);
    };

    void set(T _alpha, T _beta) {
        alpha = _alpha;
        beta = _beta;
    };

    template <typename Tnew> operator GammaNatural<Tnew>() const {
        GammaNatural<Tnew> gamma_new;
        gamma_new.alpha = alpha;
        gamma_new.beta = beta;
        gamma_new.alpha_min = alpha_min;
        gamma_new.beta_min = beta_min;
        gamma_new.alpha_max = alpha_max;
        gamma_new.beta_max = beta_max;

        return gamma_new;
    };

    VectorXT<T> encode_vector(bool unconstrained) const {
        VectorXT<T> vec(encoded_size);
        if (unconstrained) {
            vec(0) = unbox_parameter(alpha, alpha_min, alpha_max, 1.0);
            vec(1) = unbox_parameter(beta, beta_min, beta_max, 1.0);
        } else {
            vec(0) = alpha;
            vec(1) = beta;
        }
        return vec;
    }

    void decode_vector(VectorXT<T> vec, bool unconstrained) {
        if (unconstrained) {
            alpha = box_parameter(vec(0), alpha_min, alpha_max, 1.0);
            beta = box_parameter(vec(1), beta_min, beta_max, 1.0);
        } else {
            alpha = vec(0);
            beta = vec(1);
        }
    }

    void print(std::string name) {
        std::cout << "GammaNatural " << name << ": ";
        std::cout << "alpha = " << alpha << " beta = " << beta << "\n";
    }

    // Return the log likelihood of an observation obs.
    T log_lik(T obs) {
        // Stan needs us to define other variables for some reason.
        T this_alpha = alpha;
        T this_beta = beta;
        return stan::math::gamma_log<false>(obs, this_alpha, this_beta);
    }
};


std::vector<Triplet> GetMomentCovariance(GammaNatural<double>, int);


template <class T> class GammaMoments: public Parameter<T> {
private:
    void Init() {
        e = 1;
        e_log = 0;
        e_min = 0;
        encoded_size = 2;
    }
public:

    using Parameter<T>::encoded_size;

    T e;
    T e_log;

    // For unconstrained encoding.
    T e_min;

    GammaMoments() {
        Init();
    };

    GammaMoments(T _e, T _e_log) {
        Init();
        set(_e, _e_log);
    };

    void set(T _e, T _e_log) {
        e = _e;
        e_log = _e_log;
    };

    template <typename Tnew> operator GammaMoments<Tnew>() const {
        GammaMoments<Tnew> gamma_new;
        gamma_new.e = e;
        gamma_new.e_log = e_log;
        gamma_new.e_min = e_min;
        return gamma_new;
    };

    GammaMoments(GammaNatural<T> gamma_nat) {
        Init();
        e = gamma_nat.alpha / gamma_nat.beta;
        e_log = get_e_log_gamma(gamma_nat.alpha, gamma_nat.beta);
    };

    // Return the expected log likelihood up to a constant.
    T ExpectedLogLikelihood(T alpha, T beta) {
        return (alpha - 1) * e_log - beta * e;
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

    void decode_vector(VectorXT<T> vec, bool unconstrained) {
        e_log = vec(1);
        if (unconstrained) {
            e = exp(vec(0)) + e_min;
        } else {
            e = vec(0);
        }
    }

    void print(std::string name) {
        std::cout << "GammaMoments " << name << ": ";
        std::cout << "e = " << e << " e_log = " << e_log << "\n";
    }
};


//////// Wishart

template <class T> class WishartNatural: public Parameter<T> {
    void Init(int _dim) {
        dim = _dim;
        n = 0;
        v = PosDefMatrixParameter<T>(dim);
        v.mat = MatrixXT<T>::Zero(dim, dim);
        diag_min = 0;
        n_min = 0;
        encoded_size = 1 + v.size_ud;
    }
public:

    using Parameter<T>::encoded_size;
    int dim;

    PosDefMatrixParameter<T> v;
    T n;

    // For unconstrained encoding.
    T n_min;
    T diag_min;

    WishartNatural(int _dim) {
        Init(_dim);
    };

    WishartNatural() {
        Init(1);
    };

    WishartNatural(T _n, MatrixXT<T> _v) {
        Init(_v.rows());
        set(_n, _v);
    };

    void set(T _n, MatrixXT<T> _v) {
        if (_v.rows() != dim) {
            std::ostringstream msg;
            msg << "set called for wrong sized matrix. " <<
                "dim = " << dim << " size = " << _v.rows() << "\n";
            throw std::runtime_error(msg.str());
        }
        if (_v.rows() != _v.cols()) {
            throw std::runtime_error("v is not square");
        }
        n = _n;
        v.set(_v);
    };

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

    void decode_vector(VectorXT<T> vec, bool unconstrained) {
        VectorXT<T> v_vec = vec.segment(1, v.size_ud);
        if (unconstrained) {
            n = exp(vec(0)) + n_min;
            v.set_unconstrained_vec(v_vec, diag_min);
        } else {
            n = vec(0);
            v.set_vec(v_vec);
        }
    };

    void print(std::string name) {
        std::cout << "WishartNatural " << name << ":\n";
        std::cout << "n = " << n << "\n";
        PrintMatrix(v.mat, "v = ");
    }

    T log_lik(MatrixXT<T> obs) {
        // For some reason these can't be passed directly to Stan.
        T this_n = n;
        MatrixXT<T> this_v(v.mat);
        return stan::math::wishart_log(obs, this_n, this_v);
    }
};


template <class T> class WishartMoments: public Parameter<T> {
    void Init(int _dim) {
        dim = _dim;
        e_log_det = 0;
        e = PosDefMatrixParameter<T>(dim);
        e.mat = MatrixXT<T>::Zero(dim, dim);
        diag_min = 0;
        encoded_size = 1 + e.size_ud;
    }
public:
    using Parameter<T>::encoded_size;
    int dim;

    PosDefMatrixParameter<T> e;
    T e_log_det;

    // For the unconstrained parameterization.
    T diag_min;

    WishartMoments(int _dim) {
        Init(_dim);
    };

    WishartMoments() {
        Init(1);
    };

    WishartMoments(T _e_log_det, MatrixXT<T> _e) {
        Init(_e.rows());
        set(_e_log_det, _e);
    }

    void set(T _e_log_det, MatrixXT<T> _e) {
        if (_e.rows() != dim) {
            std::ostringstream msg;
            msg << "set called for wrong sized matrix. " <<
                "dim = " << dim << " size = " << _e.rows() << "\n";
            throw std::runtime_error(msg.str());
        }
        if (_e.rows() != _e.cols()) {
            throw std::runtime_error("e is not square");
        }
        e_log_det = _e_log_det;
        e.set(_e);
    };

    WishartMoments(WishartNatural<T> wishart_nat) {
        dim = wishart_nat.dim;
        Init(dim);
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

    void decode_vector(VectorXT<T> vec, bool unconstrained) {
        VectorXT<T> e_vec = vec.segment(1, e.size_ud);
        e_log_det = vec(0);
        if (unconstrained) {
            e.set_unconstrained_vec(e_vec, diag_min);
        } else {
            e.set_vec(e_vec);
        }
    };

    void print(std::string name) {
        std::cout << "WishartMoments " << name << ":\n";
        std::cout << "e_log_det = " << e_log_det << "\n";
        PrintMatrix(e.mat, "e = ");
    }

    // The expected log likelihood up to a constant.
    T ExpectedLogLikelihood(MatrixXT<T> v, T n) {
        // A different inverse might be appropriate.
        MatrixXT<T> v_inv_e = v.ldlt().solve(e.mat);
        return 0.5 * (n - dim - 1) * e_log_det - 0.5 * (v_inv_e).trace();
    }
};

std::vector<Triplet> GetMomentCovariance(WishartNatural<double>, int);


//////// Multivariate Normal


template <class T> class MultivariateNormalNatural: public Parameter<T> {
    void Init(int _dim) {
        dim = _dim;
        loc = VectorXT<T>::Zero(dim);
        info = PosDefMatrixParameter<T>(dim);
        info.mat = MatrixXT<T>::Zero(dim, dim);
        diag_min = 0;
        encoded_size = dim + info.size_ud;
    }
public:
    using Parameter<T>::encoded_size;
    int dim;
    VectorXT<T> loc;
    PosDefMatrixParameter<T> info;

    T diag_min;

    void set(VectorXT<T> _loc, MatrixXT<T> _info) {
        if (dim != _loc.size()) {
            std::ostringstream msg;
            msg << "set called for wrong sized vector. " <<
                "dim = " << dim << " size = " << loc.size() << "\n";
            throw std::runtime_error(msg.str());
        }
        if (_info.cols() != _info.rows()) {
            throw std::runtime_error("info is not square");
        }
        if (_loc.size() != _info.rows()) {
            throw std::runtime_error("loc and info have different sizes");
        }
        loc = _loc;
        info.set(_info);
    };

    MultivariateNormalNatural(int _dim) {
        Init(_dim);
    };

    MultivariateNormalNatural() {
        Init(1);
    };

    MultivariateNormalNatural(VectorXT<T> _loc, MatrixXT<T> _info) {
        Init(_loc.size());
        set(_loc, _info);
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

    void decode_vector(VectorXT<T> vec, bool unconstrained) {
        VectorXT<T> sub_vec = vec.segment(0, dim);
        loc = sub_vec;
        sub_vec = vec.segment(dim, info.size_ud);
        if (unconstrained) {
            info.set_unconstrained_vec(sub_vec, diag_min);
        } else {
            info.set_vec(sub_vec);
        }
    };

    void print(std::string name) {
        std::cout << "MultivariateNormalNatural " << name << ":\n";
        PrintVector(loc, "loc = ");
        PrintMatrix(info.mat, "info = ");
    };

    T log_lik(VectorXT<T> obs) {
        // For some reason these can't be passed directly to Stan.
        VectorXT<T> mean_vec = loc;
        MatrixXT<T> info_mat = info.mat;
        return stan::math::multi_normal_prec_log(obs, mean_vec, info_mat);
        // MatrixXT<T> sigma = info.mat.inverse();
        // return stan::math::multi_normal_log(obs, mean_vec, sigma);
    }
};


template <class T> class MultivariateNormalMoments: public Parameter<T> {
    void Init(int _dim) {
        dim = _dim;
        e_vec = VectorXT<T>::Zero(dim);
        e_outer = PosDefMatrixParameter<T>(dim);
        e_outer.mat = MatrixXT<T>::Zero(dim, dim);
        diag_min = 0;
        encoded_size = dim + e_outer.size_ud;
    }
public:
    using Parameter<T>::encoded_size;
    int dim;
    VectorXT<T> e_vec;
    PosDefMatrixParameter<T> e_outer;

    T diag_min;

    MultivariateNormalMoments(int _dim) {
        Init(_dim);
    };

    MultivariateNormalMoments() {
        Init(1);
    };

    MultivariateNormalMoments(VectorXT<T> _e_vec, MatrixXT<T> _e_outer) {
        Init(_e_vec.size());
        set(_e_vec, _e_outer);
    };

    void set(VectorXT<T> _e_vec, MatrixXT<T> _e_outer) {
        if (dim != _e_vec.size()) {
            std::ostringstream msg;
            msg << "set called for wrong sized vector. " <<
                "dim = " << dim << " size = " << e_vec.size() << "\n";
            throw std::runtime_error(msg.str());
        }
        if (_e_outer.cols() != _e_outer.rows()) {
            throw std::runtime_error("e_outer is not square");
        }
        if (_e_vec.size() != _e_outer.rows()) {
            throw std::runtime_error("e_vec and e_outer have different sizes");
        }
        e_vec = _e_vec;
        e_outer.set(_e_outer);
    };

    // Set from a vector of another type.
    template <typename Tnew> MultivariateNormalMoments(VectorXT<Tnew> mean) {
        Init(mean.size());
        e_vec = mean.template cast<T>();
        MatrixXT<T> e_vec_outer = e_vec * e_vec.transpose();
        e_outer.set(e_vec_outer);
    };

    // Set from natural parameters.
    MultivariateNormalMoments(MultivariateNormalNatural<T> mvn_nat) {
        Init(mvn_nat.dim);
        MatrixXT<T> e_outer_mat =
            mvn_nat.loc * mvn_nat.loc.transpose() + mvn_nat.info.mat.inverse();
        set(mvn_nat.loc, e_outer_mat);
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

    void decode_vector(VectorXT<T> vec, bool unconstrained) {
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
    T ExpectedLogLikelihood(MultivariateNormalMoments<T> mean,
                            WishartMoments<T> info) const {
        MatrixXT<T> mean_outer_prods =
            mean.e_vec * e_vec.transpose() + e_vec * mean.e_vec.transpose();
        return
            -0.5 * (info.e.mat * (e_outer.mat - mean_outer_prods + mean.e_outer.mat)).trace() +
            0.5 * info.e_log_det;
    };

    T ExpectedLogLikelihood(VectorXT<T> mean, WishartMoments<T> info) const {
        MatrixXT<T> mean_outer_prods = mean * e_vec.transpose() + e_vec * mean.transpose();
        MatrixXT<T> mean_outer = mean * mean.transpose();
        return
            -0.5 * (info.e.mat * (e_outer.mat - mean_outer_prods + mean_outer)).trace() +
            0.5 * info.e_log_det;
    };

    T ExpectedLogLikelihood(VectorXT<T> mean, MatrixXT<T> info) const {
        MatrixXT<T> mean_outer_prods = mean * e_vec.transpose() + e_vec * mean.transpose();
        MatrixXT<T> mean_outer = mean * mean.transpose();
        return
            -0.5 * (info * (e_outer.mat - mean_outer_prods + mean_outer)).trace() +
            0.5 * log(info.determinant());
    };

    void print(std::string name) {
        std::cout << "MultivariateNormalMoments " << name << ":\n";
        PrintVector(e_vec, "e_vec = ");
        PrintMatrix(e_outer.mat, "e_outer = ");
    };
};

std::vector<Triplet> GetMomentCovariance(MultivariateNormalNatural<double>, int);

//////// Univariate Normal

template <class T> class UnivariateNormalNatural: public Parameter<T> {
    void Init() {
        loc = 0;
        info = 0;
        info_min = 0;
        encoded_size = 2;
    }
public:
    using Parameter<T>::encoded_size;
    T loc;  // The expectation.
    T info; // The expectation of the square.

    T info_min;

    UnivariateNormalNatural() {
        Init();
    };

    UnivariateNormalNatural(T _loc, T _info) {
        Init();
        set(_loc, _info);
    };

    void set(T _loc, T _info) {
        loc = _loc;
        info = _info;
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

    void decode_vector(VectorXT<T> vec, bool unconstrained) {
        if (unconstrained) {
            loc = vec(0);
            info = exp(vec(1)) + info_min;
        } else {
            loc = vec(0);
            info = vec(1);
        }
    };

    void print(std::string name) {
        std::cout << "UnivariateNormalNatural " << name << ": ";
        std::cout << "loc = " << loc << " info = " << info << "\n";
    };

    T log_lik(T obs) {
        T mean = loc;
        T stddev = 1 / sqrt(info);
        return stan::math::normal_log(obs, mean, stddev);
    }
};


template <class T> class UnivariateNormalMoments: public Parameter<T> {
    void Init() {
        e = 0;
        e2 = 0;
        e2_min = 0;
        encoded_size = 2;
    }
public:

    using Parameter<T>::encoded_size;
    T e;  // The expectation.
    T e2; // The expectation of the square.

    T e2_min;

    UnivariateNormalMoments() {
        Init();
    };

    UnivariateNormalMoments(T _e, T _e2) {
        Init();
        set(_e, _e2);
    };

    void set(T _e, T _e2) {
        e = _e;
        e2 = _e2;
    };

    // Set from natural parameters.
    UnivariateNormalMoments(UnivariateNormalNatural<T> uvn_nat) {
        Init();
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

    void decode_vector(VectorXT<T> vec, bool unconstrained) {
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

    void print(std::string name) {
        std::cout << "UnivariateNormalMoments " << name << ":\n";
        std::cout << "e = " << e << " e2 = " << e2 << "\n";
    };
};

std::vector<Triplet> GetMomentCovariance(UnivariateNormalNatural<double>, int);


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
