import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import numpy as np
from tqdm import trange

from sklearn.linear_model import LogisticRegression
from deepx import T
from deepx.nn import *
from deepx.stats import Gaussian, Dirichlet, NIW, Categorical, kl_divergence, Bernoulli
from activations import Gaussian as GaussianLayer
from activations import GaussianStats

N = 1000
D = 10

p_w = Gaussian([T.constant(np.eye(D).astype(T.floatx()))[None], T.constant(np.zeros(D).astype(T.floatx()))[None]])

def logistic(x):
    return 1 / (1 + np.exp(-x))

# def generate_data(N, D):
    # with T.session() as s:
        # w = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D))

        # X = np.random.normal(size=(N, D))
        # p = logistic(np.einsum('ia,a->i', X, w))
        # Y = s.run(Bernoulli(p.astype(T.floatx())).sample()[0])
    # return (X, Y), w
def generate_data(N, D, sigma0=1, sigma=5, seed=None):
    K = 2
    np.random.seed(seed)
    pi = [0.5, 0.5]
    X = np.zeros((N, D))
    Y = np.zeros(N)
    mu, sigma = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D) * sigma0, size=[K]), np.tile(np.eye(D)[None] * sigma, [K, 1, 1])
    for i in range(N):
        z = Y[i] = np.random.choice(K, p=pi)
        X[i] = np.random.multivariate_normal(mean=mu[z], cov=sigma[z])
    return X, Y

def make_variable(dist):
    return dist.__class__(T.variable(T.to_float(dist.get_parameters('natural'))), parameter_type='natural')

(X, Y) = generate_data(N, D, seed=3)
cf = LogisticRegression(fit_intercept=False)
cf.fit(X, Y)
coef_ = cf.coef_
score_ = cf.score(X, Y)

q_w = make_variable(Gaussian([T.to_float(np.eye(D))[None], T.to_float(np.zeros(D))[None]]))

x, y = T.matrix(), T.vector()

lr = 1e-4
batch_size = T.shape(x)[0]
num_batches = T.to_float(N / batch_size)

with T.initialization('xavier'):
    # stats_net = Relu(D + 1, 20) >> Relu(20) >> GaussianLayer(D)
    stats_net = GaussianLayer(D + 1, D)
net_out = stats_net(T.concat([x, y[..., None]], -1))
stats = T.sum(net_out.get_parameters('natural'), 0)[None]

natural_gradient = (p_w.get_parameters('natural') + num_batches * stats - q_w.get_parameters('natural')) / N
next_w = Gaussian(q_w.get_parameters('natural') + lr * natural_gradient, parameter_type='natural')

l_w = kl_divergence(q_w, p_w)[0]

p_y = Bernoulli(T.sigmoid(T.einsum('jw,iw->ij', next_w.expected_value(), x)))
l_y = T.sum(p_y.log_likelihood(y[..., None]))
elbo = l_w + l_y

nat_op = T.assign(q_w.get_parameters('natural'), next_w.get_parameters('natural'))
grad_op = tf.train.RMSPropOptimizer(1e-4).minimize(-elbo)
train_op = tf.group(nat_op, grad_op)
sess = T.interactive_session()

predictions = T.cast(T.sigmoid(T.einsum('jw,iw->i', q_w.expected_value(), T.to_float(X))) + 0.5, np.int32)
accuracy = T.mean(T.to_float(T.equal(predictions, T.constant(Y.astype(np.int32)))))

def iter(num_iter=1, b=100):
    for _ in range(num_iter):
        idx = np.random.permutation(N)[:b]
        sess.run(train_op, {x:X[idx], y:Y[idx]})
        print("%f" % (tuple(sess.run([elbo], {x:X, y:Y}))))
    print(sess.run(q_w.get_parameters('regular')[1][0]), coef_[0])
    print(sess.run(accuracy), score_)
