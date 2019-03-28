import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

def visualize(X, y_pred, means, title):
    for i in range(len(X)):
        if y_pred[i] == 0:
            plt.scatter(X[i,0], X[i,1], c='blue', alpha=0.7)
        elif y_pred[i] == 1:
            plt.scatter(X[i,0], X[i,1], c='red', alpha=0.7)
        else:
            plt.scatter(X[i,0], X[i,1], c='gray', alpha=0.7)
    plt.scatter(means[0,0], means[0,1], c='blue', marker='x', label=f"Cluster 0 | μ: ({round(means[0,0], 2)}, {round(means[0,1], 2)})", linewidths=2, edgecolors='black')
    plt.scatter(means[1,0], means[1,1], c='red', marker='x', label=f"Cluster 1 | μ: ({round(means[1,0], 2)}, {round(means[1,1], 2)})", linewidths=2, edgecolors='black')
    plt.legend()
    plt.title(title)
    plt.show()

# Meta Parameters that can be tweaked
NUM_CLUSTERS = 2
MAX_NUM_ITERATIONS = 10
EPSILON = 0.001

# Read randomly generated dataset
X = np.genfromtxt('rand2D_2.csv', delimiter=",")
y_pred = np.array(np.ones((len(X),1)) * -1, dtype='int')

# Initialize means
means = np.array([[0,0], 
                  [1,1]], dtype="double")

# Visualize data
visualize(X, y_pred, means, "Initial")

for r in range(MAX_NUM_ITERATIONS):

    # Update classes for each data point
    for i in range(len(X)):
        m = float("inf")
        c = -1
        for k in range(NUM_CLUSTERS):
            d = norm(X[i] - means[k])
            if (d < m):
                m = d
                c = k
        y_pred[i] = c

    # Find new mean values
    new_means = np.zeros((NUM_CLUSTERS, len(X[i])))
    cluster_counts = [0 for _ in range(NUM_CLUSTERS)]

    for i in range(len(X)):
        k = int(y_pred[i])
        new_means[k] += X[i]
        cluster_counts[k] += 1

    # Update the means & evalute rate of change
    change = 0
    previous_means = np.copy(means)
    for k in range(NUM_CLUSTERS):
        if cluster_counts[k] > 0:            
            means[k] = new_means[k] / cluster_counts[k]
            change += np.sum(np.absolute(means[k] - previous_means[k]))


    # Check if convergence has been reached
    print(f"Round {r+1} | Change: {change}")
    if change < EPSILON:
        print("Convergence reached...")
        visualize(X, y_pred, means, f"Round {r+1} (Final)")
        break
    else:
        # Show updated mean
        visualize(X, y_pred, means, f"Round {r+1}")