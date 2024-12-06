import torch.nn.functional as F

#Struc2vec Imports
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import from_networkx
import torch.nn as nn
import networkx as nx

#Basic imports
import matplotlib.pyplot as plt
import random as rd
import torch


# Function to create a graph with a specific number of connected components
def create_graph_with_components(num_components, min, max):
    G = nx.Graph()
    for i in range(num_components):
        # Create a random number of nodes for the component
        num_nodes = rd.randint(min, max)
        # Create a random connected component
        component = nx.erdos_renyi_graph(n=num_nodes, p=0.4)
        while not nx.is_connected(component):
            component = nx.erdos_renyi_graph(n=num_nodes, p=0.4)
        # Relabel nodes to avoid overlaps
        mapping = {node: node + len(G) for node in component.nodes}
        component = nx.relabel_nodes(component, mapping)
        # Add the component to the main graph
        G = nx.compose(G, component)
    return G

# Visualization function
def plot_graph(G, title):
    plt.figure(figsize=(8, 6))
    nx.draw_spring(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=600, font_size=10)
    plt.title(title)
    plt.show()


def bfs_count_components(adj_list):
    visited = set()  # Track visited nodes
    num_components = 0

    def bfs(start_node):
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(neighbor for neighbor in adj_list[node] if neighbor not in visited)

    for node in adj_list:
        if node not in visited:
            num_components += 1
            bfs(node)

    return num_components

# Convert adjacency matrix to adjacency list
def adj_matrix_to_list(adj_matrix):
    adj_list = {i: [] for i in range(len(adj_matrix))}
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:  # Check if there's an edge
                adj_list[i].append(j)
    return adj_list


class Structure2Vec(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.global_pool = pyg_nn.global_mean_pool  # Use mean pooling for graph-level representation
        self.fc = nn.Linear(hidden_channels, num_classes)  # Class logits

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        graph_embedding = self.global_pool(x, batch)  # Aggregate node embeddings
        return self.fc(graph_embedding)  # Return class logits

    def message(self, x_j):
        """
        Defines the message function.
        :param x_j: Neighbor node features.
        :return: Transformed features.
        """
        return self.mlp(x_j)

    def update(self, aggr_out):
        """
        Update node features after aggregation.
        :param aggr_out: Aggregated features.
        :return: Updated node embeddings.
        """
        return self.out_proj(aggr_out)
    
# Create example graphs
graph1 = create_graph_with_components(1,15,20)  # 1 connected component
graph2 = create_graph_with_components(2,15,20)  # 2 connected components
graph3 = create_graph_with_components(3,15,20)  # 3 connected components

# Plot the graphs
plot_graph(graph1, "Graph with 1 Connected Component")
plot_graph(graph2, "Graph with 2 Connected Components")
plot_graph(graph3, "Graph with 3 Connected Components")

#Graphs with 15-20 nodes
graphs = []
for i in range(5000):
  graphs.append(create_graph_with_components(1,15,20))
for i in range(5000):
  graphs.append(create_graph_with_components(2,15,20))
for i in range(5000):
  graphs.append(create_graph_with_components(3,15,20))

# Extract adjacency matrix from custom graph object
adj_matrix = nx.to_numpy_array(graphs[0])  # Returns the adjacency matrix as a NumPy array
adj_list = adj_matrix_to_list(adj_matrix)

# Use the BFS function to count the number of connected components
num_components = bfs_count_components(adj_list)
print(f"Number of connected components: {num_components}")

graph_comp = []
for i in range(len(graphs)):
  adj_matrix = nx.to_numpy_array(graphs[i])  # Returns the adjacency matrix as a NumPy array
  adj_list = adj_matrix_to_list(adj_matrix)
  num_components = bfs_count_components(adj_list)
  graph_comp.append(num_components)


datas = []
for i in range(len(graphs)):
  label = graph_comp[i]
  data = from_networkx(graphs[i])
  data.x = torch.ones(data.num_nodes, 1)  # Set all node features to 1
  data.y = torch.tensor([label], dtype=torch.long)
  datas.append(data)

app = datas[0]
print(app.edge_index)

# # Create a DataLoader for batching
# #random.shuffle(datas)
# train_loader = DataLoader(datas[:90], batch_size=32, shuffle=True)
# test_loader = DataLoader(datas[10:], batch_size=32, shuffle=False)

# # Initialize the Structure2Vec model
# model = Structure2Vec(in_channels=datas[0].num_features, hidden_channels=32, num_classes=3)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# # Training loop
# for epoch in range(100):
#     model.train()
#     epoch_loss = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         embedding = model(batch.x, batch.edge_index, batch.batch)
#         loss = criterion(embedding, batch.y)  # Batch labels
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")


