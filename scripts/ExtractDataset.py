import cv2
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split

#
import GAT

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
print(kp1[0])

# Create graph and add nodes
G = nx.Graph()
for i in range(len(kp1)):
    G.add_node(i, pos=(kp1[i].pt[0],
    kp1[i].pt[1]), img=1)
for i in range(len(kp2)):
    G.add_node(i+len(kp1), pos=(kp2[i].pt[0], kp2[i].pt[1]), img=2)

# Add edges
for match in matches:
    i1 = match.queryIdx
    i2 = match.trainIdx + len(kp1)
    G.add_edge(i1, i2)

# Convert graph to PyTorch Geometric Data object
print(np.zeros((G.number_of_nodes(), des1.shape[1] + 2)).astype(np.float32))
print('Tensor')
print(torch.LongTensor(np.array(list(G.edges)).T))
data = Data(
    x=torch.from_numpy(np.zeros((G.number_of_nodes(), des1.shape[1] + 2)).astype(np.float32)),
    edge_index=torch.LongTensor(np.array(list(G.edges)).T),
    y=torch.LongTensor(np.zeros((G.number_of_nodes(),)))
)
pos = nx.get_node_attributes(G, 'pos')
pos_array = np.array(list(pos.values()))
pos_array[:, 0] = 2 * (pos_array[:, 0] - np.min(pos_array[:, 0])) / np.ptp(pos_array[:, 0]) - 1
pos_array[:, 1] = 2 * (pos_array[:, 1]) - np.min(pos_array[:, 1]) / np.ptp(pos_array[:, 1]) - 1
data.x[:, :-2] = torch.from_numpy(np.vstack([des1, des2]).astype(np.float32))
data.x[:, -2:] = torch.from_numpy(pos_array.astype(np.float32))
data.y[len(kp1):] = 1

# Split data into training and validation sets
train, test = train_test_split(list(data), test_size=0.2)
# print(f'Training set size: {len(train)}')
# print(f'Test set size: {len(test)}')
print(data.num_features)

## Create the model
gat = GAT.GAT(data.num_features, 8, data.num_nodes)
# print(gat)
GAT.train(gat, data)

