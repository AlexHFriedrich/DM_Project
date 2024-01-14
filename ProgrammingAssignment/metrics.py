import matplotlib.pyplot as plt
from scipy.stats import kendalltau

"""
This file contains functions for computing the approximation ratio and Kendall's tau, as well as a function for
printing the metrics.
"""


def plot_approximation_ratio(data, test_out, net, measure=""):
    """
    :param data: PyTorch Geometric Data object
    :param test_out: predicted values
    :param net: network name
    :param measure: centrality measure
    Produce a histogram of the approximation ratios
    """
    eps = 1e-8  # small constant to avoid division by 0

    approx_ratios = (test_out / (data.y.view(-1) + eps)).view(-1).detach()

    plt.hist(approx_ratios, bins=50, alpha=0.7, color='blue', label='Approximation Ratios')
    plt.axvline(x=1, color='red', linestyle='--', label='True Value')
    plt.xlabel('Approximation Ratio')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Approximation Ratios - {measure}')
    plt.legend()
    plt.savefig(f'plots/{net}_approx_ratios_{measure}.png')


def compute_kendall_tau(data, test_out):
    """
    :param data: PyTorch Geometric Data object
    :param test_out: predicted values
    Compute Kendall's tau between the true and predicted values
    """

    # Compute Kendall's tau
    tau, _ = kendalltau(data.y.detach(), test_out.detach())

    return tau

