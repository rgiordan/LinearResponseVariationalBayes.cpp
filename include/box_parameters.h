# ifndef BOX_PARAMETERS_H
# define BOX_PARAMETERS_H

# include <limits>
# include <stan/math.hpp>

using stan::math::logit;
using stan::math::inv_logit;

// template<typename T> T inv_logit(T x) {
//     if (x < 0) {
//         throw std::runtime_error("x is less than zero");
//     }
//     if (x > 1) {
//         throw std::runtime_error("x is greater than one");
//     }
//     return -1 * log(1.0 / x - 1);
// };
//
// template<typename T> T logit(T x) {
//     return 1.0 / (1.0 + exp(-1.0 * x));
// };


template<typename T>
T unbox_parameter(T x, double lower_bound, double upper_bound, double scale) {
    if (lower_bound > upper_bound) {
        throw std::runtime_error("lower_bound must be less than upper_bound.");
    }
    if (x < lower_bound) {
        throw std::runtime_error("lower_bound must be less than x.");
    }

    if (x > upper_bound) {
        throw std::runtime_error("upper_bound must be greater than x.");
    }

    // If the upper bound is infinity, just interpret it as a positivity constraint.
    if (upper_bound == std::numeric_limits<double>::infinity()) {
        return log(x - lower_bound) * scale;
    } else {
        T x_bounded = (x - lower_bound) / (upper_bound - lower_bound);
        return logit(x_bounded) * scale;
    }
}


template<typename T>
T box_parameter(T x, double lower_bound, double upper_bound, double scale) {
    if (lower_bound > upper_bound) {
        throw std::runtime_error("lower_bound must be less than upper_bound.");
    }

    // If the upper bound is infinity, just interpret it as a positivity constraint.
    if (upper_bound == std::numeric_limits<double>::infinity()) {
        return exp(x / scale) + lower_bound;
    } else {
        return inv_logit(x / scale) * (upper_bound - lower_bound) + lower_bound;
    }
}


# endif
