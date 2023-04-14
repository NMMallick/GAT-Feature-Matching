import cv2
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

# Load two images
img1 = cv2.imread('../imgs/ellie.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../imgs/ellie.jpg', cv2.IMREAD_GRAYSCALE)

# Create SIFT detector and descriptor
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Create Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x:x.distance)

# Get top k matches
k = 50
matches = matches[:k]

# Create graph and add nodes
G = nx.Graph()
for i in range(len(kp1)):
    G.add_node(i, pos=(kp1[i].pt[0], kp1[i].pt[1]), img=1, features=des1[i])
for i in range(len(kp2)):
    G.add_node(i+len(kp1), pos=(kp2[i].pt[0], kp2[i].pt[1]), img=2, features=des2[i])

# Add edges
for match in matches:
    i1 = match.queryIdx
    i2 = match.trainIdx + len(kp1)
    G.add_edge(i1, i2)

# Convert to adjacency matrix
A = nx.adjacency_matrix(G)

# Convert to tensor
A = torch.tensor(np.array(A.todense()))

# Get node features
node_features = []
for node in G.nodes():
    node_features.append(G.nodes[node]['features'])
node_features = torch.tensor(np.array(node_features))

# Get node positions
node_positions = []
for node in G.nodes():
    node_positions.append(G.nodes[node]['pos'])
node_positions = torch.tensor(node_positions)

# Convert edge_index to COO format
edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

# Create PyTorch geometric data
data = Data(x=node_features.float(), pos=node_positions.float(), edge_index=edge_index, edge_attr=None)

# Normalize node positions to be in [-1,1] range
data.pos = (data.pos - data.pos.min(dim=0).values) / (data.pos.max(dim=0).values - data.pos.min(dim=0).values) * 2 - 1