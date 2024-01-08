import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx


# read in Graph data for Pytorch Geometric
def read_graph(path):
    return nx.read_adjlist(path, delimiter=' ', nodetype=int, comments='%')


paths = ['./data/moreno_propro/out.moreno_propro_propro',
         'data/moreno_health/out.moreno_health_health']
G = read_graph(paths[0])


def calc_targets(graph):
    degree_centrality = nx.degree_centrality(graph)
    eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
    page_rank = nx.pagerank(graph)
    return [degree_centrality, eigenvector_centrality, page_rank]


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

    # train_loader = DataLoader([data], shuffle=True)

    for epoch in range(num_steps):
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.view(-1)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # print(f'Epoch {epoch + 1}/{max_iter}, Loss: {running_loss / len(train_loader)}')

    print('Training finished.')
    train_loss = criterion(out.view(-1)[data.train_mask], data.y[data.train_mask])
    print(f'Train Loss: {train_loss.item()}')
    # train_acc = torch.sum(
    #   torch.where((out.view(-1)[data.train_mask] - data.y[data.train_mask]) < tol, 1, 0)) / torch.sum(data.train_mask)
    # print(f'Train Accuracy: {train_acc.item()}')
    model.eval()
    with torch.no_grad():
        test_out = model(data.x.view(-1, 1), data.edge_index)

        test_loss = criterion(test_out.view(-1), data.y)
        print(f'Test Loss: {test_loss.item()}')

        # test_acc = torch.sum(torch.where((test_out.view(-1) - data.y) < tol, 1, 0)) / len(data.y)
        # print(f'Test Accuracy: {test_acc.item()}')


for p in paths:
    data_sets = preprocessing(p)
    for train_data in data_sets:
        model = GNN(in_channels=1, hidden_channels=128, out_channels=1)
        process(model, train_data, num_steps=500)
