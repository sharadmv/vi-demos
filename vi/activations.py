from deepx import T
from deepx.nn import Linear
from deepx import stats

__all__ = ['Gaussian', 'Bernoulli']

log1pexp = lambda x: T.log(1. + T.exp(x) + 1e-4)

class Gaussian(Linear):

    def __init__(self, *args, **kwargs):
        self.cov_type = kwargs.pop('cov_type', 'diagonal')
        super(Gaussian, self).__init__(*args, **kwargs)
        assert not self.elementwise

    def get_dim_out(self):
        return [self.dim_out[0] * 2]

    def activate(self, X):
        import tensorflow as tf
        if self.cov_type == 'diagonal':
            sigma, mu = T.split(X, 2, axis=-1)
            sigma = T.matrix_diag(log1pexp(sigma))
            return stats.Gaussian([sigma, mu], parameter_type='regular')
        raise Exception("Undefined covariance type: %s" % self.cov_type)

    def __str__(self):
        return "Gaussian(%s)" % self.dim_out

class GaussianStats(Linear):

    def __init__(self, *args, **kwargs):
        super(GaussianStats, self).__init__(*args, **kwargs)

    def get_dim_out(self):
        D = self.dim_out[0]
        return [D]

    def activate(self, X):
        shape = T.shape(X)
        return stats.NIW.pack([
            T.outer(X, X),
            X,
            T.ones(shape[:-1]),
            T.ones(shape[:-1])
        ])
        # L = T.lower_triangular(s)
        # return stats.NIW.pack([
            # T.matmul(L, L, transpose_b=True),
            # mu,
            # T.ones(shape[:-1]),
            # T.ones(shape[:-1]),
        # ])
        # s, mu = T.split(X, [D * (D + 1) // 2, D], axis=-1)
        # L = T.lower_triangular(s)
        # return stats.NIW.pack([
            # T.matmul(L, L, transpose_b=True),
            # mu,
            # T.ones(shape[:-1]),
            # T.ones(shape[:-1]),
        # ])

    def __str__(self):
        return "Gaussian(%s)" % self.dim_out

class Bernoulli(Linear):

    def __init__(self, *args, **kwargs):
        super(Bernoulli, self).__init__(*args, **kwargs)
        assert not self.elementwise

    def activate(self, X):
        return stats.Bernoulli(T.sigmoid(X), parameter_type='regular')

    def __str__(self):
        return "Bernoulli(%s)" % self.dim_out
