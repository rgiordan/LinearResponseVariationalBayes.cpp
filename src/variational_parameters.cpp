# include "variational_parameters.h"

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
