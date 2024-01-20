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

    absol_dists = (torch.abs(test_out - data.y.view(-1))).view(-1).detach()

    plt.hist(absol_dists, bins=50, alpha=0.7, color='blue', label='Absolute Distances')
    plt.axvline(x=0, color='red', linestyle='--', label='True Value')
    plt.xlabel('Absolute Distances')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of absolute distances - {measure}')
    plt.legend()
    plt.savefig(f'plots/{net}_absol_dist_{measure}.png')
    plt.close()

    plt.scatter(test_out, data.y.view(-1).detach())
    lims = [min(data.y.view(-1).detach()), max(data.y.view(-1).detach())]
    plt.plot(lims, lims, color="red", label='Identity line')
    # add a legend to the plot with legend() or plt.legend()
    plt.legend()
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title(f'Predicted vs. True Values of Node Property - {measure}')
    plt.savefig(f'plots/{net}_scatter_{measure}.png')
    plt.close()


def compute_kendall_tau(data, test_out, test_mask=None):
    """
    :param data: PyTorch Geometric Data object
    :param test_out: predicted values
    :param test_mask: mask for the test set
    Compute Kendall's tau between the true and predicted values
    """
    if test_mask is None:
        test_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    # Compute Kendall's tau
    tau, _ = kendalltau(data.y.detach()[test_mask], test_out.detach()[test_mask])

    return tau
