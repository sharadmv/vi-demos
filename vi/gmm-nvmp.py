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

with T.initialization('xavier'):
    obs_net = Relu(2, 50) >> Relu(50) >> GaussianLayer(2)
obs_net.initialize()

def generate_data(N, D, K, sigma0=10, sigma=10, seed=None):
    np.random.seed(seed)
    pi = np.random.dirichlet([100] * K)
    X = np.zeros((N, D))
    mu, sigma = np.random.multivariate_normal(mean=np.zeros(D), cov=np.eye(D) * sigma0, size=[K]), np.tile(np.eye(D)[None] * sigma, [K, 1, 1])
    Z = np.zeros(N)
    for i in range(N):
        z = Z[i] = np.random.choice(K, p=pi)
        X[i] = np.random.multivariate_normal(mean=mu[z], cov=sigma[z])
    return X.astype(np.float32), (mu, sigma, Z)
    return sess.run(obs_net(T.constant(X.astype(np.float32))).sample()[:, 0]), (mu, sigma, Z)

fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots(2, 2)
plt.ion()
plt.show()

def make_variable(dist):
    return dist.__class__(T.variable(dist.get_parameters('natural')), parameter_type='natural')

def draw():
    mean, cov, a = sess.run([mu, sigma, alpha])
    ax.cla()
    ax2.cla()
    ax2.plot(elbos)
    ax3[0][0].cla()
    ax3[0][0].plot(l_thetas)
    ax3[0][1].cla()
    ax3[0][1].plot(l_pis)
    ax3[1][0].cla()
    ax3[1][0].plot(l_zs)
    ax3[1][1].cla()
    ax3[1][1].plot(l_xs)
    ax3[0][0].set_title('theta')
    ax3[0][1].set_title('pi')
    ax3[1][0].set_title('z')
    ax3[1][1].set_title('x')
    ax.scatter(*data.T, s=1)
    ax.scatter(*sess.run(encoded_data).T, s=1)
    for k in range(K):
        plot_ellipse(a[k], mean[k], cov[k], ax=ax)
    plt.pause(0.01)
    plt.draw()

def iter(batch_size=20):
    batch_idx = np.random.permutation(N)[:batch_size]
    sess.run(train_op, {X:data[batch_idx]})
    e, ltheta, lpi, lz, lx = sess.run([elbo, l_theta, l_pi, l_z, l_x], {X:data})
    elbos.append(e)
    l_thetas.append(ltheta)
    l_pis.append(lpi)
    l_zs.append(lz)
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
K = 2
D = 2

sess = T.interactive_session()

sigma = 0.5
sigma0 = 100
data, truth = generate_data(N, D, K, sigma=sigma, sigma0=sigma0, seed=1)
p_pi = Dirichlet(T.constant(100.0 * np.ones([K], dtype=T.floatx())))
p_theta = NIW(list(map(lambda x: T.constant(np.array(x).astype(T.floatx())), [np.eye(D) * sigma, np.zeros(D), 1, D + 1])))
prior = (p_pi, p_theta)

np.random.seed(None)

X = T.placeholder(T.floatx(), [None, D])
batch_size = T.shape(X)[0]

with T.initialization('xavier'):
    net = Relu(100) >> Relu(100) >> GaussianStats(D)
encoded_data = Gaussian.unpack(net(T.constant(data)))[1]

np.set_printoptions(suppress=True)

x_tmessage = net(X)
x_tmessage2 = NIW.pack([
    T.outer(X, X),
    X,
    T.ones([batch_size]),
    T.ones([batch_size]),
])


q_pi = make_variable(Dirichlet(np.ones([K], dtype=T.floatx())))
q_theta = make_variable(NIW(map(lambda x: np.array(x).astype(T.floatx()), [np.tile(np.eye(D)[None] * 100, [K, 1, 1]), np.random.multivariate_normal(mean=np.zeros([D]), cov=np.eye(D) * 20, size=[K]), np.ones(K), np.ones(K) * (D + 1)])))
# q_theta = make_variable(NIW(map(lambda x: np.array(x).astype(T.floatx()), [truth[1] * K, truth[0], np.ones(K), np.ones(K) * (D + 1)])))

sigma, mu = Gaussian(q_theta.expected_sufficient_statistics(), parameter_type='natural').get_parameters('regular')
alpha = Categorical(q_pi.expected_sufficient_statistics(), parameter_type='natural').get_parameters('regular')

pi_cmessage = q_pi.expected_sufficient_statistics()
theta_cmessage = q_theta.expected_sufficient_statistics()

num_batches = N / T.to_float(batch_size)
nat_scale = 1.0

parent_z = q_pi.expected_sufficient_statistics()[None]
new_z = T.einsum('iab,jab->ij', x_tmessage, theta_cmessage) + parent_z
q_z = Categorical(new_z - T.logsumexp(new_z, -1)[..., None], parameter_type='natural')
p_z = Categorical(parent_z - T.logsumexp(parent_z, -1), parameter_type='natural')
l_z = T.sum(kl_divergence(q_z, p_z))
z_pmessage = q_z.expected_sufficient_statistics()

pi_stats = T.sum(z_pmessage, 0)
parent_pi = p_pi.get_parameters('natural')
current_pi = q_pi.get_parameters('natural')
pi_gradient = nat_scale / N * (parent_pi + num_batches * pi_stats - current_pi)
l_pi = T.sum(kl_divergence(q_pi, p_pi))

theta_stats = T.einsum('ia,ibc->abc', z_pmessage, x_tmessage)
theta_stats2 = T.einsum('ia,ibc->abc', z_pmessage, x_tmessage2)
parent_theta = p_theta.get_parameters('natural')[None]
current_theta = q_theta.get_parameters('natural')
theta_gradient = nat_scale / N * (parent_theta + num_batches * theta_stats - current_theta)
new_theta = NIW(parent_theta + num_batches * theta_stats, parameter_type='natural')
l_theta = T.sum(kl_divergence(q_theta, p_theta))

x_param = T.einsum('ia,abc->ibc', z_pmessage, new_theta.expected_sufficient_statistics())
q_x = Gaussian(x_param, parameter_type='natural')
l_x = T.sum(q_x.log_likelihood(X))

elbo = l_theta + l_pi + l_x + l_z
elbos = []
l_thetas = []
l_pis = []
l_zs = []
l_xs = []

natgrads = [(theta_gradient, q_theta.get_parameters('natural')),
            (pi_gradient, q_pi.get_parameters('natural'))]

# nat_op = tf.group(*[T.assign(b, a + b) for a, b in natgrads])
nat_opt = tf.train.MomentumOptimizer(1e-2, 0.9)
nat_op = nat_opt.apply_gradients([(-a, b) for a, b in natgrads])
grad_op = tf.train.AdamOptimizer(1e-3).minimize(-elbo, var_list=net.get_parameters())

train_op = tf.group(grad_op, nat_op)

sess.run(tf.global_variables_initializer())

draw()
for i in trange(1000000):
    iter(20)
    if i % 1000 == 0:
        draw()
draw()
