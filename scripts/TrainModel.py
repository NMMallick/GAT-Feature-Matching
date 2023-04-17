# from torch_geometric.data import DataLoader, Dataset, Data

# from torchvision.transforms import ToTensor
# import networkx as nx
# import numpy as np
# import torch
# import cv2

# class SIFTGraphDataset(Dataset):
#     def __init__(self, img_paths, labels):
#         self.img_paths = img_paths
#         self.labels = labels
#         self.transforms = ToTensor()

#     def __getitem__(self, idx):
#         # Load image
#         img_path = self.img_paths[idx]
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#         # Extract SIFT features
#         sift = cv2.SIFT_create()
#         kp, des = sift.detectAndCompute(img, None)

#         # Create graph and add nodes
#         G = nx.Graph()
#         for i in range(len(kp)):
#             G.add_node(i, pos=(kp[i].pt[0], kp[i].pt[1]), img=idx, features=des[i])

#         # Add edges
#         if idx == 0:
#             # For the first image, add self-loops
#             for i in range(len(kp)):
#                 G.add_edge(i, i)
#         else:
#             # For other images, match features with the first image
#             img1 = cv2.imread(self.img_paths[0], cv2.IMREAD_GRAYSCALE)
#             kp1, des1 = sift.detectAndCompute(img1, None)
#             bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#             matches = bf.match(des, des1)
#             matches = sorted(matches, key=lambda x:x.distance)
#             k = min(50, len(matches))
#             matches = matches[:k]
#             for match in matches:
#                 i1 = match.queryIdx
#                 i2 = match.trainIdx
#                 G.add_edge(i1, i2)

#         # Convert to adjacency matrix
#         A = nx.adjacency_matrix(G)

#         # Convert to tensor
#         A = torch.tensor(A.todense())

#         # Get node features
#         node_features = []
#         for node in G.nodes():
#             node_features.append(G.nodes[node]['features'])
#         node_features = torch.tensor(node_features)

#         # Get node positions
#         node_positions = []
#         for node in G.nodes():
#             node_positions.append(G.nodes[node]['pos'])
#         node_positions = torch.tensor(node_positions)

#         edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)


#         # Create PyTorch geometric data object
#         data = Data(x=node_features.float(), pos=node_positions.float(), edge_index=edge_index, edge_attr=None)

#         # Normalize node positions to be in [-1,1] range
#         data.pos = (data.pos - data.pos.min(dim=0).values) / (data.pos.max(dim=0).values - data.pos.min(dim=0).values) * 2 - 1

#         # Apply transforms to data
#         data = self.transforms(data)

#         # Get label
#         label = torch.tensor(self.labels[idx])

#         return data, label

#     def __len__(self):
#         return len(self.img_paths)


############################################################3

import torch
import torch.optim as optim
from sklearn.metrics import f1_score

def train(model, optimizer, criterion, features, adj, labels, idx_train, idx_val, epochs):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = criterion(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        val_loss, val_acc, val_f1 = evaluate(model, criterion, features, adj, labels, idx_val)
        print("Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f}"
              .format(epoch+1, epochs, train_loss, val_loss, val_acc, val_f1))

def evaluate(model, criterion, features, adj, labels, idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss = criterion(output[idx_val], labels[idx_val])
        preds = output.argmax(dim=1)
        acc = (preds[idx_val] == labels[idx_val]).sum().item() / idx_val.shape[0]
        f1 = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy(), average='micro')
        return loss.item(), acc, f1

# Training hyperparameters
lr = 0.005
weight_decay = 5e-4
n_epochs = 200
patience = 20

# Create the GAT model
nfeat = features.shape[1]
nhid = 8
nclass = labels.max().item() + 1
dropout = 0.6
alpha = 0.2
nheads = 8

model = GAT(nfeat, nhid, nclass, dropout, alpha, nheads)

# Create optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# Train the model
train(model, optimizer, criterion, features, adj, labels, idx_train, idx_val, n_epochs)
