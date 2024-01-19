import matplotlib.pyplot as plt
import torch
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
    Produce a histogram of the absolute distances between the true and predicted values
    """

    approx_ratios = (torch.abs(test_out - data.y.view(-1))).view(-1).detach()

    plt.hist(approx_ratios, bins=50, alpha=0.7, color='blue', label='Absolute Distances')
    plt.axvline(x=0, color='red', linestyle='--', label='True Value')
    plt.xlabel('Absolute Distances')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of absolute distances - {measure}')
    plt.legend()
    plt.savefig(f'plots/{net}_approx_ratios_{measure}.png')
    plt.close()

    plt.scatter(test_out, data.y.view(-1).detach())
    plt.show()


def compute_kendall_tau(data, test_out):
    """
    :param data: PyTorch Geometric Data object
    :param test_out: predicted values
    Compute Kendall's tau between the true and predicted values
    """

    # Compute Kendall's tau
    tau, _ = kendalltau(data.y.detach(), test_out.detach())

    return tau
