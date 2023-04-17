# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GATLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, dropout, alpha, concat=True):
#         super(GATLayer, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.dropout = dropout
#         self.alpha = alpha
#         self.concat = concat

#         self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)

#         self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def forward(self, h, adj):
#         Wh = torch.mm(h, self.W)
#         a_input = self._prepare_attentional_mechanism_input(Wh)
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))

#         # masking out the unconnected nodes
#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)

#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)

#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime

#     def _prepare_attentional_mechanism_input(self, Wh):
#         N = Wh.size()[0]
#         Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
#         Wh_repeated_alternating = Wh.repeat(N, 1)
#         all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
#         return all_combinations_matrix.view(N, N, 2*self.out_dim)

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         self.attentions = [GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = GATLayer(nhid*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)


# import torch.nn.functional as F
# import torch
# from torch.nn import Linear, Dropout
# from torch_geometric.nn import GCNConv, GATv2Conv


# class GCN(torch.nn.Module):
#   """Graph Convolutional Network"""
#   def __init__(self, dim_in, dim_h, dim_out):
#     super().__init__()
#     self.gcn1 = GCNConv(dim_in, dim_h)
#     self.gcn2 = GCNConv(dim_h, dim_out)
#     self.optimizer = torch.optim.Adam(self.parameters(),
#                                       lr=0.01,
#                                       weight_decay=5e-4)

#   def forward(self, x, edge_index):
#     h = F.dropout(x, p=0.5, training=self.training)
#     h = self.gcn1(h, edge_index)
#     h = torch.relu(h)
#     h = F.dropout(h, p=0.5, training=self.training)
#     h = self.gcn2(h, edge_index)
#     return h, F.log_softmax(h, dim=1)


# class GAT(torch.nn.Module):
#   """Graph Attention Network"""
#   def __init__(self, dim_in, dim_h, dim_out, heads=8):
#     super().__init__()
#     self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
#     self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
#     self.optimizer = torch.optim.Adam(self.parameters(),
#                                       lr=0.005,
#                                       weight_decay=5e-4)

#   def forward(self, x, edge_index):
#     h = F.dropout(x, p=0.6, training=self.training)
#     h = self.gat1(x, edge_index)
#     h = F.elu(h)
#     h = F.dropout(h, p=0.6, training=self.training)
#     h = self.gat2(h, edge_index)
#     return h, F.log_softmax(h, dim=1)

# def accuracy(pred_y, y):
#     """Calculate accuracy."""
#     return ((pred_y == y).sum() / len(y)).item()

# def train(model, data):
#     """Train a GNN model and return the trained model."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = model.optimizer
#     epochs = 200
#     print(data.train_mask)
#     model.train()
#     for epoch in range(epochs+1):
#         # Training
#         optimizer.zero_grad()
#         _, out = model(data.x, data.edge_index)
#         loss = criterion(out[data.train_mask], data.y[data.train_mask])
#         acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()

#         # Validation
#         val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
#         val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

#         # Print metrics every 10 epochs
#         if(epoch % 10 == 0):
#             print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
#                   f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
#                   f'Val Acc: {val_acc*100:.2f}%')

#     return model

# def test(model, data):
#     """Evaluate the model on test set and print the accuracy score."""
#     model.eval()
#     _, out = model(data.x, data.edge_index)
#     acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
#     return acc

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

class GAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.w = torch.nn.Parameter(torch.Tensor(num_heads, in_dim, out_dim))
        self.a = torch.nn.Parameter(torch.Tensor(num_heads, 2*out_dim, 1))
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        torch.nn.init.xavier_uniform_(self.w, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, x):
        h = torch.stack([torch.mm(x, self.w[i]) for i in range(self.num_heads)])
        a_input = torch.cat([h[i].unsqueeze(1).repeat(1, x.shape[0], 1) for i in range(self.num_heads)], dim=1)
        a_output = torch.cat([h[i].unsqueeze(2).repeat(1, 1, x.shape[0]) for i in range(self.num_heads)], dim=0)
        a_input = a_input.view(self.num_heads, x.shape[0]*x.shape[0], -1)
        a_output = a_output.view(self.num_heads, x.shape[0]*x.shape[0], -1)
        a = self.leaky_relu(torch.matmul(torch.cat([a_input, a_output], dim=2), self.a))
        a = F.softmax(a, dim=1)
        output = torch.matmul(a.transpose(1, 2), h.transpose(0, 1)).squeeze().transpose(0, 1)
        return output

if __name__== '__main__':
    sift_features_image1 = np.load("features/descriptors.npy")
    sift_features_image2 = np.load("features/descriptors.npy")

    k = 10  # number of nearest neighbors to consider
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(sift_features_image2)
    distances, indices = nbrs.kneighbors(sift_features_image1)


    adjacency_matrix = np.zeros((sift_features_image1.shape[0], sift_features_image2.shape[0]))
    for i in range(sift_features_image1.shape[0]):
        for j in indices[i]:
            adjacency_matrix[i, j] = 1

    adjacency_matrix = torch.from_numpy(adjacency_matrix).float()

    x1 = torch.from_numpy(sift_features_image1).float()
    x2 = torch.from_numpy(sift_features_image2).float()


    model = GAT(in_dim=x2.shape[1], out_dim=128, num_heads=8)
    output1 = model(x1)
    output2 = model(x2)

