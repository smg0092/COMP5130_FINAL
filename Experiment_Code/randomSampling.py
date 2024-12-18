import torch
import math
import random
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GATConv


# Change this number to change how many nodes you want to attack
NODES_TARGETTED = 15
attacked_nodes = []

dataset = Planetoid(root='data/Planetoid', name='Citeseer', transform=NormalizeFeatures())
#dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
#dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

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

def train(model, data, optimizer, criterion):
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

def test(model, data):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc

def randSamp(inputData, model):
    dataSample = inputData.clone()
    hitlist = []

    # Get results before modification
    model.eval()
    ouch = model(dataSample.x, dataSample.edge_index)
    ogPredictions = ouch.argmax(dim=1)

    # Append only correctly predicted nodes to hitlist
    candidate = random.randint(0, len(ogPredictions) - 1)
    while candidate in hitlist or ogPredictions[candidate] != dataSample.y[candidate]:
        candidate = random.randint(0, len(ogPredictions) - 1)
    hitlist.append(candidate)
    print("Hitlist: ", hitlist)

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
            return None
        
        # Get new results
        model.eval()
        ywouch = model(dataSample.x, dataSample.edge_index)
        newPredictions = ywouch.argmax(dim=1)

        # Check constraints
        constraint = constraintSatisfaction(dataSample, ogPredictions, newPredictions, hitlist)
        target_down = targetDown(ogPredictions, newPredictions, hitlist)

        # If constraints not met, restore backup
        if not constraint:
            dataSample = backupData.clone()
            stopping += 1
            print("Reverting to backup")
            if(stopping > 10):
                return None
        else:
            backupData = dataSample.clone()
            stopping = 0
    print("Subtracting edges done")
    attacked_nodes.append(hitlist[0])
    return dataSample

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

# Check if the constraints are satisfied
# If the node is not in the hitlist, the correct predictions should not change
# If the node was previously attacked, it should remain an incorrect prediction
def constraintSatisfaction(inData, ogPredictions, newPredictions, hitlist):
    for i in range(len(ogPredictions)):
        if i not in hitlist:
            if i in attacked_nodes:
                if ogPredictions[i] == inData.y[i] and ogPredictions[i] == newPredictions[i]:
                    return False
            elif ogPredictions[i] == inData.y[i] and ogPredictions[i] != newPredictions[i]:
                return False
    return True

def targetDown(ogPredictions, newPredictions, hitlist):
  for i in range(len(hitlist)):
    if ogPredictions[hitlist[i]] == newPredictions[hitlist[i]]:
      return False
  return True

# Hamming distance between two predictions
# Prints the indexes that are different
def hamming_distance(pred1, pred2):
    return (pred1 != pred2).sum().item()

def runRandSamp(numNodes, datasetNum):
    match datasetNum:
        case 1:
            dataset = Planetoid(root='data/Planetoid', name='Citeseer', transform=NormalizeFeatures())
        case 2:
            dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
        case 3:
            dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

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


    model = GCN(hidden_channels=16)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        loss = train(model, data, optimizer, criterion)
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    #Get results before attacking the graph
    test_acc = test(model, data)
    print(f'Test Accuracy: {test_acc:.4f}')
    model.eval()
    out = model(data.x, data.edge_index)
    oldPred = out.argmax(dim=1)
    dataOld = data.clone()
    attacked_nodes = []

    for i in range(numNodes):
        dataNew = randSamp(data, model)
        while dataNew == None:
            dataNew = randSamp(data, model)
        data = dataNew

    #Results after attacking the graph
    test_acc = test(model, data)
    model.eval()
    out = model(data.x, data.edge_index)
    newPred = out.argmax(dim=1)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Hamming distance: {hamming_distance(oldPred, newPred)}')
    print("Edges removed: ", dataOld.edge_index.shape[1] - data.edge_index.shape[1])