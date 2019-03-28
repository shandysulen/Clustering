import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm, det, inv
from math import exp, sqrt, pi
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def assign_cluster(x, means, sigmas):
    max_p = 0
    cluster = -1
    for k in range(NUM_CLUSTERS):
        p = get_prob(x, means, sigmas, k)
        if p > max_p:
            max_p = p
            cluster = k
    return cluster

def batch_gaussian(pos, u, sigma):
    v = np.einsum('...k,kl,...l->...', pos - u, inv(sigma), pos - u)
    return np.exp(-0.5 * v) / np.sqrt(pow(2 * pi, 2) * det(sigma))

def calc_covariance_mat(data, mean):
    sigma = np.zeros((2,2))
    for x in data:
        sigma += np.outer(x - mean, x - mean)
    return sigma / len(data)

def calc_R(X, means, pis, sigmas):
    R = np.zeros((len(X), NUM_CLUSTERS))
    for i in range(len(X)):
        for k in range(NUM_CLUSTERS):
            denom = 0
            for l in range(NUM_CLUSTERS):
                denom += pis[l] * gaussian(X[i], means[l], sigmas[l])
            R[i][k] = pis[k] * gaussian(X[i], means[k], sigmas[k]) / denom
    return R

def gaussian(x, u, sigma):
    return exp(-0.5 * (x-u).T @ inv(sigma) @ (x-u)) / sqrt(pow(2 * pi, 2) * det(sigma))

def get_prob(x, means, sigmas, k):
    num = gaussian(x, means[k], sigmas[k])
    denom = 0

    for l in range(NUM_CLUSTERS):
        denom += gaussian(x, means[l], sigmas[l])
    
    return num / denom


def update_mean(X, R, k):
    num = [0, 0]
    for i in range(len(X)):
        num += R[i,k] * X[i]
    val = num / sum(R[:,k])
    return val

def update_occupancy(X, means, sigmas, pis, k):
    s = 0
    for i in range(len(X)):
        denom = 0
        for l in range(NUM_CLUSTERS):
            denom += pis[l] * gaussian(X[i], means[l], sigmas[l])
        s += (pis[k] * gaussian(X[i], means[k], sigmas[k])) / denom
    
    return s / len(X)
        
def update_sigma(X, y, mean, k):
    sigma = np.zeros((2,2))
    c = 0
    for i in range(len(X)):
        if y[i] == k:
            sigma += np.outer(X[i] - mean, X[i] - mean)
            c += 1
    return sigma / c
    

def visualize(X, means, sigmas, title):

    # Create grid
    N = 100
    x = np.linspace(-2, 4, N)
    y = np.linspace(-2, 4, N)
    xx, yy = np.meshgrid(x, y)

    # Pack X and Y into single 3-dimensional array
    pos = np.empty(xx.shape + (2,))
    pos[:,:,0] = xx
    pos[:,:,1] = yy

    # Plot data points
    for i in range(len(X)):
        if y_pred[i] == 0:
            plt.scatter(X[i,0], X[i,1], c='blue', alpha=0.7)
        elif y_pred[i] == 1:
            plt.scatter(X[i,0], X[i,1], c='red', alpha=0.7)
        else:
            plt.scatter(X[i,0], X[i,1], c='gray', alpha=0.7)
    plt.scatter(means[0,0], means[0,1], c='blue', marker='x', label=f"Cluster 0 | μ: ({round(means[0,0], 3)}, {round(means[0,1], 3)}), |Σ|: {round(det(sigmas[0]), 2)}")
    plt.scatter(means[1,0], means[1,1], c='red', marker='x', label=f"Cluster 0 | μ: ({round(means[1,0], 3)}, {round(means[1,1], 3)}), |Σ|: {round(det(sigmas[1]), 2)}")
    plt.legend()

    # Plot Gaussians 
    Z0 = batch_gaussian(pos, means[0], sigmas[0])
    Z1 = batch_gaussian(pos, means[1], sigmas[1])
    plt.contour(xx, yy, Z0, cmap='Blues')
    plt.contour(xx, yy, Z1, cmap='Reds')
    plt.title(title)
    plt.show()

# Meta Parameters that can be tweaked
NUM_CLUSTERS = 2
MAX_NUM_ITERATIONS = 10
EPSILON = 0.001

# Read randomly generated dataset
X = np.genfromtxt('rand2D.csv', delimiter=",")
y_pred = np.array(np.ones((len(X),1)) * -1, dtype='int')

# Initialize pi, means, and sigmas
pis = [(1/NUM_CLUSTERS) for _ in range(NUM_CLUSTERS)]
means = np.array([[0,0], 
                  [1,1]], dtype="double")
sigmas = np.array([calc_covariance_mat(X[:50,:], means[0]), calc_covariance_mat(X[50:100,:], means[1])])

# Create predicted class label vector
y_pred = np.ones((len(X), 1)) * -1
for i in range(len(X)):
    y_pred[i] = assign_cluster(X[i], means, sigmas)

# Visualize initial start
visualize(X, means, sigmas, "Initial")

for r in range(MAX_NUM_ITERATIONS):

    # Update R
    R = calc_R(X, means, pis, sigmas)

    # Archive previous means
    previous_means = np.copy(means)

    for k in range(NUM_CLUSTERS):

        # Update Means & calculate change
        means[k] = update_mean(X, R, k)

        # Update Sigmas
        sigmas[k] = update_sigma(X, y_pred, means[k], k)

        # Updates Pis
        pis[k] = update_occupancy(X, means, sigmas, pis, k)
    
    # Update y_pred
    for j in range(len(X)):
        y_pred[j] = assign_cluster(X[j], means, sigmas)
    
    # Visualize initial start
    visualize(X, means, sigmas, f"Round {r+1}")

    # Calculate change
    change = np.sum(np.abs(means - previous_means))

    # Output
    print(f"Round {r+1} | Change: {change}")
    if change < EPSILON:
        print("Convergence reached...")
        break


