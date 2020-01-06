import numpy as np
import pymc3 as pm
import theano.tensor as tt

from astroML.linear_model import LinearRegression


__all__ = ['LinearRegressionwithErrors']


class LinearRegressionwithErrors(LinearRegression):

    def fit(self, X, y, y_error, x_error,
            sample_kwargs={'draws': 1000, 'target_accept': 0.9}):

        X = np.atleast_2d(X)
        x_error = np.atleast_2d(x_error)

        with pm.Model():
            # slope and intercept of eta-ksi relation
            slope = pm.Flat('slope', shape=(X.shape[0], ))
            inter = pm.Flat('inter')

            # intrinsic scatter of eta-ksi relation
            int_std = pm.HalfFlat('int_std')
            # standard deviation of Gaussian that ksi are drawn from (assumed mean zero)
            tau = pm.HalfFlat('tau', shape=(X.shape[0],))
            # intrinsic ksi
            mu = pm.Normal('mu', mu=0, sd=tau, shape=(X.shape[0],))

            # Some wizzarding with the dimensions all around.
            ksi = pm.Normal('ksi', mu=mu, tau=tau, shape=X.T.shape)

            # intrinsic eta-ksi linear relation + intrinsic scatter
            eta = pm.Normal('eta', mu=(tt.dot(slope.T, ksi.T) + inter),
                            sd=int_std, shape=y.shape)

            # observed xi, yi
            x = pm.Normal('xi', mu=ksi.T, sd=x_error, observed=X, shape=X.shape)
            y = pm.Normal('yi', mu=eta, sd=y_error, observed=y, shape=y.shape)

            self.trace = pm.sample(**sample_kwargs)

        return self
