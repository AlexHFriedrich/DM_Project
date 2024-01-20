import time
from typing import Any, Union

import networkit as nk
import networkit.centrality as nkc
import numpy as np
import torch
import torch_geometric
from networkit.centrality import PageRank, EigenvectorCentrality, DegreeCentrality
from torch_geometric.data import Data
from torch_geometric.utils import from_networkit

"""
This file handles the reading of the data from the KONECT database, calculation of the centrality measures using 
networkit and the conversion of the data to PyTorch Geometric Data objects.
"""

paths = {"Adolescent health network": "data/moreno_health/out.moreno_health_health",
         "Yeast network": "data/moreno_propro/out.moreno_propro_propro"}


def path_to_data(path) -> tuple[dict[str, tuple[Data, Any]], dict[str, Any]]:
    """
    :param path: path to the graph
    :return: a tuple containing a dictionary of the data sets and a dictionary of the running times of the networkit
    """
    # read graph
    konectReader = nk.graphio.KONECTGraphReader()
    Gnk = konectReader.read(path)

    measures, running_times_networkit = centrality_from_network_it(Gnk)
    data_sets = {}

    for measure in measures:
        data_sets[measure] = pyg_data_from_network_it(Gnk, measures[measure])

    return data_sets, running_times_networkit


def normalize_scores(scores):
    """
    :param scores: list of scores
    :return: normalized scores
    """
    scores = np.array(scores)
    scores = (scores - min(scores)) / (max(scores) - min(scores))
    return scores


def centrality_from_network_it(graph) -> tuple[
    dict[str, Union[PageRank, EigenvectorCentrality, DegreeCentrality]], dict[str, Any]]:
    """
    :param graph: networkit graph
    :return: a tuple containing a dictionary of the centrality measures and a dictionary of the running times of the
    networkit
    """
    running_times = {}
    # degree
    start = time.perf_counter()
    d = nkc.DegreeCentrality(graph, outDeg=False)
    d.run()
    running_times["Degree"] = time.perf_counter() - start
    # normalize the scores to be between 0 and 1
    d_scores = normalize_scores(d.scores())
    # eigenvector centrality
    start = time.perf_counter()
    e = nkc.EigenvectorCentrality(graph)
    e.run()
    running_times["Eigenvector Centrality"] = time.perf_counter() - start
    # normalize the scores to be between 0 and 1
    e_scores = normalize_scores(e.scores())
    # pagerank
    start = time.perf_counter()
    p = nkc.PageRank(graph, normalized=True)
    p.run()
    running_times["PageRank"] = time.perf_counter() - start
    # normalize the scores to be between 0 and 1
    p_scores = normalize_scores(p.scores())
    return {"Degree": d_scores, "Eigenvector Centrality": e_scores, "PageRank": p_scores}, running_times


def pyg_data_from_network_it(graph, measure):
    """
    :param graph: networkit graph
    :param measure: centrality measure
    :return: a PyTorch Geometric Data object
    """
    indices, weights = from_networkit(graph)
    x = torch.tensor(list(range(graph.numberOfNodes())), dtype=torch.float).view(-1, 1)
    if graph.isWeighted():
        G = torch_geometric.data.Data(x=x, edge_index=indices, weight=weights,
                                      y=torch.tensor(measure, dtype=torch.float).view(-1, 1))
    else:
        G = torch_geometric.data.Data(x=x, edge_index=indices,
                                      y=torch.tensor(measure, dtype=torch.float).view(-1, 1))

    train_mask = torch.zeros(G.num_nodes, dtype=torch.bool)
    train_mask[:int(0.9 * G.num_nodes)] = 1
    train_mask = train_mask[torch.randperm(G.num_nodes)]

    return G, train_mask


def load_data_list():
    """
    :return: a dictionary of the data sets
    """
    return {path: path_to_data(paths[path]) for path in paths}


if __name__ == "__main__":
    pass
