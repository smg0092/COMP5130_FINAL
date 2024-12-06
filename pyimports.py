#GNN Imports
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

#Struc2vec Imports
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import torch.nn as nn
import networkx as nx

#Custom Imports
from Genetic_Functions_gnn import evaluate_member_fitness
from Memeber_gnn import Member
import Struct2Vec

#Basic imports
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import time
import torch
import math




