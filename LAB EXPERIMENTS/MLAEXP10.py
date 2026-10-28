# Expectation-Maximization Algorithm

import numpy as np

# Dataset
X = np.array([1, 2, 3, 4, 8, 9, 10, 11])

# Initial parameters
mu1 = 2.0
mu2 = 9.0
sigma1 = 1.0
sigma2 = 1.0
pi1 = 0.5
pi2 = 0.5

# Gaussian probability function
def gaussian(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * \
           np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# EM Algorithm
for iteration in range(10):

    # E-step
    p1 = pi1 * gaussian(X, mu1, sigma1)
    p2 = pi2 * gaussian(X, mu2, sigma2)

    total = p1 + p2
    r1 = p1 / total
    r2 = p2 / total

    # M-step
    N1 = np.sum(r1)
    N2 = np.sum(r2)

    mu1 = np.sum(r1 * X) / N1
    mu2 = np.sum(r2 * X) / N2

    sigma1 = np.sqrt(np.sum(r1 * (X - mu1) ** 2) / N1)
    sigma2 = np.sqrt(np.sum(r2 * (X - mu2) ** 2) / N2)

    pi1 = N1 / len(X)
    pi2 = N2 / len(X)

    print("Iteration:", iteration + 1)
    print("Mean 1:", round(mu1, 3))
    print("Mean 2:", round(mu2, 3))
    print("Weight 1:", round(pi1, 3))
    print("Weight 2:", round(pi2, 3))
    print()

# Final results
print("Final Results")
print("Cluster 1 Mean:", round(mu1, 3))
print("Cluster 2 Mean:", round(mu2, 3))
# Plot the final Gaussian distributions
import matplotlib.pyplot as plt

plt.hist(X, bins=8, density=True, alpha=0.5,
         label="Dataset")

x_values = np.linspace(0, 12, 200)

y1 = pi1 * gaussian(x_values, mu1, sigma1)
y2 = pi2 * gaussian(x_values, mu2, sigma2)

plt.plot(x_values, y1, label="Cluster 1")
plt.plot(x_values, y2, label="Cluster 2")

plt.xlabel("Data Values")
plt.ylabel("Probability Density")
plt.title("Expectation-Maximization Algorithm")
plt.legend()
plt.show()
