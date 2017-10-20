import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import numpy as np
from tqdm import trange

from deepx import T
from deepx.stats import Gaussian, Dirichlet, NIW, Categorical, kl_divergence

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
plt.ion()
plt.show()

def make_variable(dist):
    return dist.__class__(T.variable(dist.get_parameters('natural')), parameter_type='natural')

def draw():
    mean, cov, a = sess.run([mu, sigma, alpha])
    ax.cla()
    ax2.cla()
    ax2.plot(elbos)
    ax.scatter(*data.T, s=1)
    for k in range(K):
        plot_ellipse(a[k], mean[k], cov[k])
    plt.pause(0.01)
    plt.draw()

def iter(batch_size=6):
    batch_idx = np.random.permutation(N)[:batch_size]
    sess.run(grad_op, {X:data[batch_idx]})
    e = sess.run(elbo, {X:data})
    elbos.append(e)


def plot_ellipse(alpha, mean, cov, line=None):
    t = np.linspace(0, 2*np.pi, 100) % (2*np.pi)
    circle = np.vstack((np.sin(t), np.cos(t)))
    ellipse = 2.*np.dot(np.linalg.cholesky(cov), circle) + mean[:,None]
    if line:
        line.set_data(ellipse)
        line.set_alpha(alpha)
    else:
        ax.plot(ellipse[0], ellipse[1], linestyle='-', linewidth=2)

N = 1000
K = 5
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

q_pi = make_variable(Dirichlet(np.ones([K], dtype=T.floatx())))
q_theta = make_variable(NIW(map(lambda x: np.array(x).astype(T.floatx()), [np.tile(np.eye(D)[None] * 100, [K, 1, 1]), np.random.multivariate_normal(mean=np.zeros([D]), cov=np.eye(D) * 20, size=[K]), np.ones(K), np.ones(K) * (D + 1)])))

sigma, mu = Gaussian(q_theta.expected_sufficient_statistics(), parameter_type='natural').get_parameters('regular')
alpha = Categorical(q_pi.expected_sufficient_statistics(), parameter_type='natural').get_parameters('regular')

pi_cmessage = q_pi.expected_sufficient_statistics()
x_tmessage = NIW.pack([
    T.outer(X, X),
    X,
    T.ones([batch_size]),
    T.ones([batch_size]),
])
x_stats = Gaussian.pack([
    T.outer(X, X),
    X,
])
theta_cmessage = q_theta.expected_sufficient_statistics()

num_batches = N / T.to_float(batch_size)
nat_scale = 10.0

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
parent_theta = p_theta.get_parameters('natural')[None]
current_theta = q_theta.get_parameters('natural')
theta_gradient = nat_scale / N * (parent_theta + num_batches * theta_stats - current_theta)
l_theta = T.sum(kl_divergence(q_theta, p_theta))

x_param = T.einsum('ia,abc->ibc', q_z.expected_sufficient_statistics(), q_theta.expected_sufficient_statistics())
q_x = Gaussian(x_param, parameter_type='natural')
l_x = T.sum(q_x.log_likelihood(X))

elbo = l_theta + l_pi + l_z + l_x
elbos = []

natgrads = [(theta_gradient, q_theta.get_parameters('natural')),
            (pi_gradient, q_pi.get_parameters('natural'))]

# grad_op = tf.group(*[T.assign(b, a + b) for a, b in natgrads])
opt = tf.train.MomentumOptimizer(1e-1, 0.9)
grad_op = opt.apply_gradients([(-a, b) for a, b in natgrads])

sess = T.interactive_session()

draw()
for i in trange(10000):
    iter()
    if i % 100 == 0:
        draw()
