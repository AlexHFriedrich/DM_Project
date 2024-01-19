import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv

"""
This file contains the GIN model class that is used for task 3. 
"""


class GIN(torch.nn.Module):
    """
    This class implements a GIN model with the given number of layers and hidden dimension.
    """
    def __init__(self, dim_h, num_layers=3):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(1, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.convolutional_layers = [GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU())) for _ in range(num_layers - 1)]

        self.linear = Linear(dim_h, dim_h)

        self.linear2 = Linear(dim_h, 1)

    def forward(self, x, edge_index):
        # Node embeddings
        h = self.conv1(x=x, edge_index=edge_index)

        for conv in self.convolutional_layers:
            h = conv(x=h, edge_index=edge_index)

        h4 = self.linear(h).relu()

        return self.linear2(h4)


if __name__ == "__main__":
    pass
