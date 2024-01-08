import networkx as nx
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
from scipy.stats import kendalltau
import pandas as pd


# read in Graph data for Pytorch Geometric
def read_graph(path):
    return nx.read_adjlist(path, delimiter=' ', nodetype=int, comments='%')


paths = ['./data/moreno_propro/out.moreno_propro_propro',
         'data/moreno_health/out.moreno_health_health']
G = read_graph(paths[0])

# Initialize an empty dictionary to store the running times
running_times = {}

def calc_targets(graph):
    centrality_measures = ['degree_centrality', 'eigenvector_centrality', 'page_rank']
    functions = [nx.degree_centrality, nx.eigenvector_centrality, nx.pagerank]
    targets = []
    
    for measure, function in zip(centrality_measures, functions):
        start_time = time.time()
        if measure == 'eigenvector_centrality':
            result = function(graph, max_iter=1000)
        else:
            result = function(graph)
        running_time = time.time() - start_time

        # Store the running time in the dictionary
        running_times[measure] = running_time

        targets.append(result)

    return targets


def create_dataset(g, targets):
    # Extract node indices and corresponding degree centrality values
    x, y = zip(*[(node, targets[node]) for node in g.nodes])

    # Convert x and y to torch tensors
    x = torch.tensor(list(x), dtype=torch.float32)
    y = torch.tensor(list(y), dtype=torch.float32)

    # Reshape x to a column vector
    x = x.view(-1, 1)

    # Convert x to a NumPy array before using MinMaxScaler
    x_array = x.numpy().reshape(-1, 1)

    # Use MinMaxScaler to normalize x to the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_normalized = scaler.fit_transform(x_array)

    # Convert the normalized NumPy array back to a PyTorch tensor
    x_normalized = torch.from_numpy(x_normalized).view(-1, 1)
    networkx_data_torch = from_networkx(g)
    return Data(x=x_normalized, edge_index=networkx_data_torch.edge_index, y=y)


def create_mask(train_data):
    random_integers = np.random.randint(0, len(train_data.y), size=int((len(train_data.y) / 10)))
    train_mask = torch.full_like(train_data.y, True, dtype=bool)
    train_mask[random_integers] = False
    train_data.train_mask = train_mask


def preprocessing(path):
    g = read_graph(path)

    targets = calc_targets(g)

    G_networkx = nx.Graph()
    G_networkx.add_edges_from(g.edges)
    train_data_sets = [create_dataset(G_networkx, target) for target in targets]

    for data_set in train_data_sets:
        create_mask(data_set)

    return train_data_sets

def plot_approximation_ratio(data, test_out):
    eps = 1e-8  # small constant to avoid division by 0
    approx_ratios = test_out.view(-1) / (data.y + eps)
    plt.hist(approx_ratios.numpy(), bins=50, alpha=0.7, color='blue', label='Approximation Ratios')
    plt.axvline(x=1, color='red', linestyle='--', label='True Value')
    plt.xlabel('Approximation Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Approximation Ratios')
    plt.legend()
    plt.show()

def compute_kendall_tau(data, test_out):
    # Compute the rankings of the true and predicted values
    true_ranking = np.argsort(data.y.numpy())
    predicted_ranking = np.argsort(test_out.view(-1).numpy())

    # Compute Kendall's tau
    tau, p_value = kendalltau(true_ranking, predicted_ranking)

    print(f"Kendall's tau: {tau}")

# Define a model
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        output = self.conv1(x, edge_index)
        output = torch.relu(output)
        output = self.conv2(output, edge_index)
        return output


def process(model, data, num_steps=100, lr=0.01):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_steps):
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.view(-1)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Training finished.')
    train_loss = criterion(out.view(-1)[data.train_mask], data.y[data.train_mask])
    print(f'Train Loss: {train_loss.item()}')
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        test_out = model(data.x.view(-1, 1), data.edge_index)
        gnn_time = time.time() - start_time
        running_times['gnn_method'] = gnn_time

        test_loss = criterion(test_out.view(-1), data.y)
        print(f'Test Loss: {test_loss.item()}')

        plot_approximation_ratio(data, test_out)
        compute_kendall_tau(data, test_out)

def print_running_times():
    df = pd.DataFrame(running_times, index=['Running Time'])
    print(df)

for p in paths:
    data_sets = preprocessing(p)
    for train_data in data_sets:
        model = GNN(in_channels=1, hidden_channels=128, out_channels=1)
        process(model, train_data, num_steps=500)
        print_running_times()

