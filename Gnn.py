from pyimports import *

dataset = Planetoid(root='data/Planetoid', name='Citeseer', transform=NormalizeFeatures())
#dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
#dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

print('======================')
print(f'Dataset: {dataset}')
print(f'Data type:')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print('======================')

data = dataset[0]  # Get the first graph object.
print("Data type of data graph object: ")
print(type(data))
print(data)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc

def randSamp(inputData, nodes_attacked):
    dataSample = inputData.clone()
    hitlist = []

    # Get results before modification
    model.eval()
    ouch = model(dataSample.x, dataSample.edge_index)
    ogPredictions = ouch.argmax(dim=1)

    # Take random nodes and add to hitlist
    # i = 0
    # while i < nodes_attacked:
    #     candidate = random.randint(0, dataSample.num_nodes - 1)
    #     if candidate not in hitlist and (ogPredictions[candidate] == dataSample.y[candidate]):
    #         hitlist.append(candidate)
    #         i += 1
    hitlist.append(1000)
    print(hitlist)

    stopping = 0
    constraint = False
    target_down = False
    backupData = dataSample.clone()
    addBackupData = dataSample.clone()
    # First, try subtracting edges from graph to satisfy constraints
    while not (constraint and target_down):
        # Remove random edge connected to hitlist nodes
        dataSample = remove_random_edge(dataSample, hitlist)
        if(dataSample == None):
            dataSample = addBackupData.clone()
            break

        # Get new results
        model.eval()
        ywouch = model(dataSample.x, dataSample.edge_index)
        newPredictions = ywouch.argmax(dim=1)

        # Check different
        if(isDifferent(ogPredictions, newPredictions)):
          print("Different")

        # Check constraints
        constraint = constraintSatisfaction(ogPredictions, newPredictions, hitlist)
        target_down = targetDown(ogPredictions, newPredictions, hitlist)

        # If constraints not met, restore backup
        if not constraint:
            dataSample = backupData.clone()
            stopping -= 1
        else:
            backupData = dataSample.clone()
            stopping += 1
    # If subtracting edges doesn't work, try additing edges
    while not (constraint and target_down):
        # Add random edge connected to hitlist nodes
        dataSample = add_random_edge(dataSample, hitlist)
        if(dataSample == None):
            dataSample = addBackupData.clone()
            break

        # Get new results
        model.eval()
        ywouch = model(dataSample.x, dataSample.edge_index)
        newPredictions = ywouch.argmax(dim=1)

        # Check different
        if(isDifferent(ogPredictions, newPredictions)):
          print("Different")

        # Check constraints
        constraint = constraintSatisfaction(ogPredictions, newPredictions, hitlist)
        target_down = targetDown(ogPredictions, newPredictions, hitlist)

        # If constraints not met, restore backup
        if not constraint:
            dataSample = backupData.clone()
            stopping -= 1
        else:
            backupData = dataSample.clone()
            stopping += 1

    return dataSample

def isDifferent(og, new):
  for i in range(len(og)):
    if og[i] != new[i]:
      return True
  return False

def add_random_edge(data, hitlist):
    # Get candidate nodes for edge addition
    candidate_nodes = []
    #find the neighbors of the hitlist nodes
    for i in hitlist:
        for j in range(data.edge_index.shape[1]):
            if data.edge_index[0][j] == i:
                #add index of the neighbor to the candidate nodes
                candidate_nodes.append(data.edge_index[1][j])
            elif data.edge_index[1][j] == i:
                candidate_nodes.append(data.edge_index[0][j])
    # Remove duplicates
    candidate_nodes = list(set(candidate_nodes))

    # Select random node to add edge
    node_to_add_edge = candidate_nodes[random.randint(0, len(candidate_nodes) - 1)]

    # Add edge between the node and a random neighbor
    # Get all neighbors of the node
    neighbors = []
    for i in range(data.edge_index.shape[1]):
        if data.edge_index[0][i] == node_to_add_edge:
            neighbors.append(data.edge_index[1][i])
        elif data.edge_index[1][i] == node_to_add_edge:
            neighbors.append(data.edge_index[0][i])

    # Select random neighbor
    neighbor_to_add_edge = neighbors[random.randint(0, len(neighbors) - 1)]

    # Add the edge
    new_edge = torch.tensor([[node_to_add_edge, neighbor_to_add_edge],
                              [neighbor_to_add_edge, node_to_add_edge]], dtype=torch.long)
    data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

    return data

def remove_random_edge(data, hitlist):
    # Get candidate nodes for edge removal
    candidate_nodes = []
    #find the neighbors of the hitlist nodes
    for i in hitlist:
        for j in range(data.edge_index.shape[1]):
            if data.edge_index[0][j] == i:
                #add index of the neighbor to the candidate nodes
                candidate_nodes.append(data.edge_index[1][j])
            elif data.edge_index[1][j] == i:
                candidate_nodes.append(data.edge_index[0][j])
    # Remove duplicates
    candidate_nodes = list(set(candidate_nodes))

    # Create list of every edge connected to the candidate nodes
    candidate_edges = []
    for i in candidate_nodes:
        for j in range(data.edge_index.shape[1]):
            if data.edge_index[0][j] == i:
                candidate_edges.append(j)
            elif data.edge_index[1][j] == i:
                candidate_edges.append(j)

    # Remove duplicates
    candidate_edges = list(set(candidate_edges))
    print(candidate_edges)
    if len(candidate_edges) == 0:
        return None

    # Select random edge to remove
    edge_to_remove = candidate_edges[random.randint(0, len(candidate_edges) - 1)]

    # Remove the edge
    # Create a mask to select all edges except the one to remove
    num_edges = data.edge_index.shape[1]
    mask = torch.ones(num_edges, dtype=torch.bool)
    mask[edge_to_remove] = False

    # Update edge_index with the remaining edges
    data.edge_index = data.edge_index[:, mask]

    return data

def constraintSatisfaction(ogPredictions, newPredictions, hitlist):
  for i in range(len(ogPredictions)):
    if i not in hitlist and (ogPredictions[i] != newPredictions[i]):
      return False
  return True

def targetDown(ogPredictions, newPredictions, hitlist):
  for i in range(len(hitlist)):
    if ogPredictions[hitlist[i]] == newPredictions[hitlist[i]]:
      return False
  return True

model = GCN(hidden_channels=16)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    loss = train()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
data = randSamp(data, 1)

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
