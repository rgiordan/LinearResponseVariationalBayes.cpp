# include "variational_parameters.h"


std::vector<Triplet> GetMomentCovariance(GammaNatural<double> nat, int offset) {
    // The offsets must match encode_vector in the moments class.
    std::vector<Triplet> terms =
        get_gamma_covariance_terms(nat.alpha, nat.beta, offset + 0, offset + 1);
    return terms;
};


std::vector<Triplet> GetMomentCovariance(WishartNatural<double> nat, int offset) {
    // The offsets must match encode_vector in the moments class.
    // In WishartMoments, e_log_det is first, followed by e.
    std::vector<Triplet> terms =
        get_wishart_covariance_terms(
            nat.v.mat, nat.n, offset + 1, offset + 0);
    return terms;
};


std::vector<Triplet>
GetMomentCovariance(MultivariateNormalNatural<double> nat, int offset) {
    // The offsets must match encode_vector in the moments class.
    // In MultivariateNormalMoments, e_vec is first, followed by e_outer.
    MatrixXd e_outer = nat.loc * nat.loc.transpose() + nat.info.mat.inverse();
    std::vector<Triplet> terms =
        get_mvn_covariance_terms(
            nat.loc, e_outer, offset + 0, offset + nat.loc.size());
    return terms;
};


std::vector<Triplet>
GetMomentCovariance(UnivariateNormalNatural<double> nat, int offset) {
    // The offsets must match encode_vector in the moments class.
    // In UnivariateNormalMoments, e is first, followed by e2.
    std::vector<Triplet> terms =
        get_normal_covariance_terms(
            nat.loc, nat.info, offset + 0, offset + 1);
    return terms;
};



# if INSTANTIATE_VARIATIONAL_PARAMETERS_H
  template class PosDefMatrixParameter<double>;
  template class PosDefMatrixParameter<var>;
  template class PosDefMatrixParameter<fvar>;

  template class GammaNatural<double>;
  template class GammaNatural<var>;
  template class GammaNatural<fvar>;

  template class GammaMoments<double>;
  template class GammaMoments<var>;
  template class GammaMoments<fvar>;

  template class WishartNatural<double>;
  template class WishartNatural<var>;
  template class WishartNatural<fvar>;

  template class WishartMoments<double>;
  template class WishartMoments<var>;
  template class WishartMoments<fvar>;

  template class MultivariateNormalNatural<double>;
  template class MultivariateNormalNatural<var>;
  template class MultivariateNormalNatural<fvar>;

  template class MultivariateNormalMoments<double>;
  template class MultivariateNormalMoments<var>;
  template class MultivariateNormalMoments<fvar>;

  template class UnivariateNormalNatural<double>;
  template class UnivariateNormalNatural<var>;
  template class UnivariateNormalNatural<fvar>;

  template class UnivariateNormalMoments<double>;
  template class UnivariateNormalMoments<var>;
  template class UnivariateNormalMoments<fvar>;
# endif
