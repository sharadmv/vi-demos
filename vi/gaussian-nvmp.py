import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import numpy as np
from tqdm import trange

from deepx import T
from deepx.nn import *
from deepx.stats import Gaussian, Dirichlet, NIW, Categorical, kl_divergence
from activations import Gaussian as GaussianLayer
from activations import GaussianStats

def generate_data(N, D, K, sigma0=10, sigma=10, seed=None):
    np.random.seed(seed)
    pi = np.random.dirichlet([100] * K)
    X = np.zeros((N, D))
    mu, sigma = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D) * sigma0, size=[K]), np.tile(np.eye(D)[None] * sigma, [K, 1, 1])
    for i in range(N):
        z = np.random.choice(K, p=pi)
        X[i] = np.random.multivariate_normal(mean=mu[z], cov=sigma[z])
    return X.astype(np.float32)

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots(2)
plt.ion()
plt.show()

def make_variable(dist):
    return dist.__class__(T.variable(dist.get_parameters('natural')), parameter_type='natural')

def draw():
    mean, cov = sess.run([mu, sigma], {X:data})
    ax.cla()
    ax2.cla()
    ax2.plot(elbos)
    ax3[0].cla()
    ax3[0].plot(l_thetas)
    ax3[1].cla()
    ax3[1].plot(l_xs)
    ax3[0].set_title('theta')
    ax3[1].set_title('x')
    ax.scatter(*data.T, s=1)
    ax.scatter(*sess.run(encoded_data).T, s=1)
    plot_ellipse(1, mean, cov, ax=ax)
    plt.pause(0.01)
    plt.draw()

def iter(batch_size=20):
    batch_idx = np.random.permutation(N)[:batch_size]
    sess.run(train_op, {X:data[batch_idx]})
    e, ltheta, lx = sess.run([elbo, l_theta, l_x], {X:data})
    elbos.append(e)
    l_thetas.append(ltheta)
    l_xs.append(lx)


def plot_ellipse(alpha, mean, cov, line=None, ax=None):
    t = np.linspace(0, 2*np.pi, 100) % (2*np.pi)
    circle = np.vstack((np.sin(t), np.cos(t)))
    ellipse = 2.*np.dot(np.linalg.cholesky(cov), circle) + mean[:,None]
    if line:
        line.set_data(ellipse)
        line.set_alpha(alpha)
    else:
        ax.plot(ellipse[0], ellipse[1], linestyle='-', linewidth=2, alpha=alpha)

N = 1000
K = 1
D = 2

sigma = 0.5
sigma0 = 100
data = generate_data(N, D, K, sigma=sigma, sigma0=sigma0, seed=None)
p_pi = Dirichlet(T.constant(10.0 * np.ones([K], dtype=T.floatx())))
p_theta = NIW(list(map(lambda x: T.constant(np.array(x).astype(T.floatx())), [np.eye(D) * sigma, np.zeros(D), 1, D + 1])))
prior = (p_pi, p_theta)

np.random.seed(None)

X = T.placeholder(T.floatx(), [None, D])
batch_size = T.shape(X)[0]

with T.initialization('xavier'):
    net = Relu(5) >> Relu(5) >> GaussianStats(D)

encoded_data = Gaussian.unpack(net(T.constant(data)))[1]

np.set_printoptions(suppress=True)

x_tmessage = net(X)
# x_tmessage = NIW.pack([
    # T.outer(X, X),
    # X,
    # T.ones([batch_size]),
    # T.ones([batch_size]),
# ])


# q_theta = make_variable(NIW(map(lambda x: np.array(x).astype(T.floatx()), [np.eye(D), np.random.multivariate_normal(mean=np.zeros([D]), cov=np.eye(D) * 20), 1.0, 1.0])))


num_batches = N / T.to_float(batch_size)
nat_scale = 1.0

theta_stats = T.sum(x_tmessage, 0)
parent_theta = p_theta.get_parameters('natural')
q_theta = NIW(parent_theta + theta_stats, parameter_type='natural')
sigma, mu = Gaussian(q_theta.expected_sufficient_statistics(), parameter_type='natural').get_parameters('regular')
# theta_gradient = nat_scale / N * (parent_theta + num_batches * theta_stats - current_theta)
l_theta = T.sum(kl_divergence(q_theta, p_theta))

x_param = q_theta.expected_sufficient_statistics()[None]
q_x = Gaussian(T.tile(x_param, [batch_size, 1, 1]), parameter_type='natural')
l_x = T.sum(q_x.log_likelihood(X))

elbo = l_theta + l_x
elbos = []
l_thetas = []
l_xs = []

# natgrads = [(theta_gradient, q_theta.get_parameters('natural'))]

# nat_op = tf.group(*[T.assign(b, a + b) for a, b in natgrads])
# nat_opt = tf.train.GradientDescentOptimizer(1e-2)
# nat_op = nat_opt.apply_gradients([(-a, b) for a, b in natgrads])
grad_op = tf.train.AdamOptimizer(1e-2).minimize(-l_x, var_list=net.get_parameters())

train_op = tf.group(grad_op)

sess = T.interactive_session()
# sess.run(T.assign(net.get_parameters()[0], T.eye(D)))

draw()
for i in trange(1000000):
    iter(N)
    if i % 1000 == 0:
        draw()
draw()
