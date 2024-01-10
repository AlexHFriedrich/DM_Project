import networkx as nx
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.nn import ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, Sequential, Linear
from torch_geometric.utils.convert import from_networkx
from scipy.stats import kendalltau
import pandas as pd


# read in Graph data for Pytorch Geometric
def read_graph(path):
    """
    :param path: path to the graph
    :return: a networkx graph
    """
    # Read in the graph
    return nx.read_adjlist(path, delimiter=' ', nodetype=int, comments='%')


def calc_targets(graph):
    """
    :param graph:
    :return: a list of targets
    """
    centrality_measures = ['degree_centrality', 'eigenvector_centrality', 'page_rank']
    functions = [nx.degree_centrality, nx.eigenvector_centrality, nx.pagerank]
    targets = []

    for measure, function in zip(centrality_measures, functions):
        start_time = time.time()

        if measure == 'eigenvector_centrality':
            if len(list(graph.edges)[0]) == 3:
                print(graph.weight)
                result = function(graph, weight='weight', max_iter=1000)
            else:
                result = function(graph, max_iter=1000)
        elif measure == 'page_rank':
            if len(list(graph.edges)[0]) == 3:
                result = function(graph, weight='weight')
            else:
                result = function(graph)
        else:
            result = function(graph)
        running_time = time.time() - start_time

        # Store the running time in the dictionary
        running_times[measure] = running_time

        targets.append(result)

    return targets


def create_dataset(g, targets):
    """
    :param g: networkx graph
    :param targets: dictionary of centrality measures
    :return: a PyTorch Geometric Data object
    """
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
    '''
    :param train_data: PyTorch Geometric Data object
    Set training mask to 90% of the data
    '''
    random_integers = np.random.randint(0, len(train_data.y), size=int((len(train_data.y) / 10)))
    train_mask = torch.full_like(train_data.y, True, dtype=bool)
    train_mask[random_integers] = False
    train_data.train_mask = train_mask


def preprocessing(path):
    """
    :param path: path to the graph
    :return: a list of PyTorch Geometric Data objects
    """
    g = read_graph(path)

    G_networkx = nx.Graph()
    G_networkx.add_edges_from(g.edges)

    if len(list(g.edges)[0]) == 3:
        print("weighted")
        G_networkx = nx.read_weighted_edgelist(path, delimiter=' ', nodetype=int, comments='%')

    if nx.is_directed(G_networkx):
        G_networkx = nx.read_edgelist(path, delimiter=' ', nodetype=int, comments='%')

    targets = calc_targets(G_networkx)

    train_data_sets = [create_dataset(G_networkx, target) for target in targets]

    for data_set in train_data_sets:
        create_mask(data_set)

    return train_data_sets


def plot_approximation_ratio(data, test_out, measure=""):
    """
    :param data: PyTorch Geometric Data object
    :param test_out: predicted values
    :param measure: centrality measure
    Produce a histogram of the approximation ratios
    """
    eps = 1e-8  # small constant to avoid division by 0
    approx_ratios = test_out.view(-1) / (data.y + eps)
    plt.hist(approx_ratios.numpy(), bins=50, alpha=0.7, color='blue', label='Approximation Ratios')
    plt.axvline(x=1, color='red', linestyle='--', label='True Value')
    plt.xlabel('Approximation Ratio')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Approximation Ratios - {measure}')
    plt.legend()
    plt.show()


def compute_kendall_tau(data, test_out):
    """
    :param data: PyTorch Geometric Data object
    :param test_out: predicted values
    Compute Kendall's tau between the true and predicted values
    """
    # Compute Kendall's tau
    tau, _ = kendalltau(data.y.numpy(), test_out.view(-1).numpy())

    print(f"Kendall's tau: {tau}")


class GIN0(torch.nn.Module):
    """
    https://github.com/sw-gong/GNN-Tutorial/blob/master/GNN-tutorial-solution.ipynb
    """
    def __init__(self, dataset, num_layers, hidden):
        super(GIN0, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
            train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                    train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GIN(torch.nn.Module):
    """
    https://github.com/sw-gong/GNN-Tutorial/blob/master/GNN-tutorial-solution.ipynb
    """
    def __init__(self, dataset, num_layers, hidden):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                    train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GNN_3(nn.Module):
    """
    A simple GNN with three GCNConv layers
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_3, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


class GNN_2(nn.Module):
    '''
    A simple GNN with two GCNConv layers
    '''

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_2, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        output = self.conv1(x, edge_index)
        output = torch.relu(output)
        output = self.conv2(output, edge_index)
        return output


def process(model, data, num_steps=100, lr=0.01, tune=False, measure_int=-1):
    """
    :param model: GNN model
    :param data: PyTorch Geometric Data object
    :param num_steps: number of training steps
    :param lr: learning rate

    Train the model and evaluate it on the test set
    """
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # store train and test error in a dictionary
    if tune:
        errors = {'test_error': []}

    for epoch in range(num_steps):
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.view(-1)[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if not tune:
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
        if tune:
            errors['test_error'].append(test_loss.item())
        else:
            print(f'Test Loss: {test_loss.item()}')
            if measure_int >= 0:
                measure = list(running_times.keys())[measure_int]

            plot_approximation_ratio(data, test_out, measure)
            compute_kendall_tau(data, test_out)
    if tune:
        return errors


def print_running_times(measure):
    """
    Print the running times of the different methods
    """
    df = pd.DataFrame(running_times, index=['Running Time'])
    cols = df.columns
    print(df[[cols[measure], cols[-1]]])


def create_model(config, model_type=GNN_2):
    """
    :param config: dictionary of model parameters
    :param model_type: GNN model
    :return: GNN model
    """
    m = model_type(config['in_channels'], config['hidden_channels'], config['out_channels'])
    return m


if __name__ == '__main__':
    # Set the random seed for reproducibility
    torch.manual_seed(0)
    paths = ['./data/moreno_propro/out.moreno_propro_propro',
             'data/moreno_health/out.moreno_health_health']

    # Initialize an empty dictionary to store the running times
    running_times = {}

    # create a variety of different models with differing hidden sizes for each of GNN_2 and GNN_3
    configs = [{'in_channels': 1, 'hidden_channels': 16, 'out_channels': 1},
               {'in_channels': 1, 'hidden_channels': 32, 'out_channels': 1},
               {'in_channels': 1, 'hidden_channels': 64, 'out_channels': 1},
               {'in_channels': 1, 'hidden_channels': 128, 'out_channels': 1},
               {'in_channels': 1, 'hidden_channels': 256, 'out_channels': 1},
               {'in_channels': 1, 'hidden_channels': 512, 'out_channels': 1}]

    models_2 = [(create_model(config, model_type=GNN_2), config) for config in configs]
    models_3 = [(create_model(config, model_type=GNN_3), config) for config in configs]

    models = models_2 + models_3

    num_steps = 50
    for p in paths:
        data_sets = preprocessing(p)
        for i, train_data in enumerate(data_sets):
            # store train and test errors of different configs in a dictionary
            errors = {'test_error': []}
            for model in models:
                res = process(model[0], train_data, num_steps=num_steps, tune=True)
                errors['test_error'].append(res['test_error'])

            # choose the model with the lowest test error
            best_model = models[np.argmin(errors['test_error'])]
            if best_model[0].__class__.__name__ == 'GNN_2':
                print('GNN_2')
            else:
                print('GNN_3')
            print(best_model[1])

            process(best_model[0], train_data, num_steps=num_steps, measure_int=i)
            print_running_times(measure=i)
