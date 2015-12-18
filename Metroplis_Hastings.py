# INFO 521 Final Assignment Option A
# Inferring parameters of a 3D line from a noisy 2D image using Metropolis Hastings
# By Meng Jia


import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


t = np.genfromtxt('data/inputs.csv', dtype=None)
points1 = np.genfromtxt('data/points_2d.csv', dtype=None, delimiter=',')
points2 = np.genfromtxt('data/points_2d_2.csv', dtype=None, delimiter=',')

camera_pos1 = np.zeros((3, 1))
camera_pos2 = np.transpose([[-5, 0, 5]])

I = np.identity(3)
M = np.concatenate((I, camera_pos1), axis=1)  # camera matrix 1

I_prime = np.fliplr(np.diag([1, 1, -1]))
M_prime = np.concatenate((I_prime, camera_pos2), axis=1)  # camera matrix 2

mu_l = np.array([0, 0, 4])  # 3D line parameters
sigma_l = np.dot(10, I)

cov1 = np.cov(points1.T)  # covariance matrix for rs distribution
cov2 = np.cov(points2.T)

n = 10000  # iterations


def start_point():
    """first sample of random walk, sampling from prior"""
    start = np.random.multivariate_normal(mu_l, sigma_l, (1, 2))

    return start


def convert_to_2d(p, M):
    """convert 3D point to 2D point"""
    step_1 = np.matrix(np.insert(p, 3, 1)).T  # insert 1 to 3D point
    u_v_w = M * step_1  # multiply new 3D point with camera matrix M
    u_v = np.concatenate((u_v_w[0], u_v_w[1]))
    q = np.divide(u_v, u_v_w[2])  # 2D point

    return q


def proposal(p):
    """proposal distribution; proposing a point based on the previous point"""
    si = np.dot(0.05**2, np.identity(3))
    prop = np.random.multivariate_normal(p, si)

    return prop


def log_prior(pi, pf):
    """log scale of prior"""
    y_i = multivariate_normal.logpdf(pi, mu_l, sigma_l)
    y_f = multivariate_normal.logpdf(pf, mu_l, sigma_l)

    return y_i + y_f


def data_likeli(pi, pf, r, M, cov):
    """log scale of likelihood"""
    log_likeli = 0

    qi = convert_to_2d(pi, M)  # convert 3D points to 2D
    qf = convert_to_2d(pf, M)

    for t_s, r_s in zip(t, r):  # read 2D point r_s and input t_s
        q_s = qi + np.dot(qf - qi, t_s)
        mu = np.array(q_s).flatten()
        y_s = multivariate_normal.logpdf(r_s, mu, cov)
        log_likeli += y_s

    return log_likeli


def calcu_posterior(pi, pf, r, M, cov):
    """log scale of posterior"""
    return log_prior(pi, pf) + data_likeli(pi, pf, r, M, cov)


def metropolis_hastings(r, M, cov):
    """Metropolis Hastings Algorithm"""

    # sample the start 3D points p_i, p_f
    pi_star, pf_star = start_point()[:,0].flatten(), start_point()[:,1].flatten()

    samples = np.zeros((n+1, 6))  # initialize sampler
    samples[0][:3] = pi_star
    samples[0][3:] = pf_star

    cur_log_prob = calcu_posterior(pi_star, pf_star, r, M, cov)  # current log probability of posterior

    acc_count = 0  # sample acceptance rate

    for i in range(1, n+1):

        if not i % 500:
            print 'Iteration %i' % i

        new_pi = proposal(samples[i-1][:3])  # propose new samples
        new_pf = proposal(samples[i-1][3:])
        new_log_prob = calcu_posterior(new_pi, new_pf, r, M, cov)  # new log probability of posterior

        ratio = new_log_prob - cur_log_prob  # acceptance ratio
        alpha = min(np.log(1), ratio)

        u = np.random.rand()

        if np.log(u) <= alpha:
            # accept new samples
            samples[i][:3] = new_pi
            samples[i][3:] = new_pf
            cur_log_prob = new_log_prob
            acc_count += 1
        else:
            # reject new samples
            samples[i] = samples[i-1]

    acc_rate = acc_count / float(n)

    return samples, acc_rate


def monte_2d(pi, pf, t_pre, cov):
    """Monte Carlo estimate of 2D output point"""
    qi = convert_to_2d(pi, M)
    qf = convert_to_2d(pf, M)
    qs = qi + np.dot(qf - qi, t_pre)
    rs = np.random.multivariate_normal(np.array(qs).flatten(), cov)

    return rs


# -------------
# MAIN FUNCTION
# -------------

mh_sample1, accu1 = metropolis_hastings(points1, M, cov1)
mh_sample2, accu2 = metropolis_hastings(points2, M_prime, cov2)


print '\nCalculating MAP and Monte Carlo estimate...\n'
R = np.zeros((n, 2))
P1 = []
P2 = []
for i in range(1, n+1):
    R[i-1] = monte_2d(mh_sample1[i][:3], mh_sample1[i][3:], 1.5, cov1)
    prob1 = calcu_posterior(mh_sample1[i][:3], mh_sample1[i][3:], points1, M, cov1)
    prob2 = calcu_posterior(mh_sample2[i][:3], mh_sample2[i][3:], points2, M_prime, cov2)

    P1.append(prob1)
    P2.append(prob2)


# Question 2
max_idx1, max_value1 = max(enumerate(set(P1)))
MAP1 = mh_sample1[max_idx1]
l1 = euclidean(MAP1[:3], MAP1[3:])
print '\nThe MAP estimate of first 3D line length (euclidean) is %s\n' % l1


# Question 3
monte_mean = np.mean(mh_sample1, axis=0)
print 'The Monte Carlo estimate of the mean of the posterior distribution: p_i = %s^T, p_f = %s^T \n' \
      % (monte_mean[:3], monte_mean[3:])


# Question 4
predicted_rs = np.mean(R, axis=0)
print 'The Monte Carlo estimate of the predicted (2D) output point at test input t = 1.5 is r = %s^T \n' \
      % predicted_rs


# Question 5
max_idx2, max_value2 = max(enumerate(set(P2)))
MAP2 = mh_sample1[max_idx2]
l2 = euclidean(MAP2[:3], MAP2[3:])
print 'The MAP estimate of second 3D line length (euclidean) is %s\n' % l2


# Question 2 plots
for i in range(6):
    plt.figure(i)
    b = mh_sample1[:,i]

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(n+1), b)
    plt.xlabel('Samples')
    plt.ylabel('value of sampled p%i' % (i+1))

    plt.subplot(2, 1, 2)
    plt.hist(b, bins=80)
    plt.xlabel('value of sampled p%i' % (i+1))
    plt.ylabel('observations')

plt.show()
