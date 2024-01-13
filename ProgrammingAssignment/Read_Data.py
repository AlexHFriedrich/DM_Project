import time
from typing import Any, Union

import networkit as nk
import networkit.centrality as nkc
import torch
import torch_geometric
from networkit.centrality import PageRank, EigenvectorCentrality, DegreeCentrality
from torch_geometric.data import Data
from torch_geometric.utils import from_networkit

paths = {"social": "data/moreno_health/out.moreno_health_health",
         "protein": "data/moreno_propro/out.moreno_propro_propro"}


def path_to_data(path) -> tuple[dict[str, tuple[Data, Any]], dict[str, Any]]:
    # read graph
    konectReader = nk.graphio.KONECTGraphReader()
    Gnk = konectReader.read(path)

    measures, running_times_networkit = centrality_from_network_it(Gnk)
    data_sets = {}

    for measure in measures:
        data_sets[measure] = pyg_data_from_network_it(Gnk, measures[measure])

    return data_sets, running_times_networkit


def centrality_from_network_it(graph) -> tuple[
    dict[str, Union[PageRank, EigenvectorCentrality, DegreeCentrality]], dict[str, Any]]:
    running_times = {}
    # degree centrality
    start = time.perf_counter()
    d = nkc.DegreeCentrality(graph, outDeg=True)
    d.run()
    running_times["Degree Centrality"] = time.perf_counter() - start
    # eigenvector centrality
    start = time.perf_counter()
    e = nkc.EigenvectorCentrality(graph)
    e.run()
    running_times["Eigenvector Centrality"] = time.perf_counter() - start

    # pagerank
    start = time.perf_counter()
    p = nkc.PageRank(graph)
    p.run()
    running_times["PageRank"] = time.perf_counter() - start

    return {"Degree Centrality": d, "Eigenvector Centrality": e, "PageRank": p}, running_times


def pyg_data_from_network_it(graph, measure):
    indices, weights = from_networkit(graph)
    x = torch.tensor(list(range(graph.numberOfNodes())), dtype=torch.float).view(-1, 1)
    if graph.isWeighted():
        G = torch_geometric.data.Data(x=x, edge_index=indices, weight=weights,
                                      y=torch.tensor(measure.scores(), dtype=torch.float).view(-1, 1))
    else:
        G = torch_geometric.data.Data(x=x, edge_index=indices,
                                      y=torch.tensor(measure.scores(), dtype=torch.float).view(-1, 1))

    train_mask = torch.zeros(G.num_nodes, dtype=torch.bool)
    train_mask[:int(0.9 * G.num_nodes)] = 1
    train_mask = train_mask[torch.randperm(G.num_nodes)]

    return G, train_mask


def load_data_list():
    return {path: path_to_data(paths[path]) for path in paths}


def main():
    return load_data_list()


if __name__ == "__main__":
    main()
