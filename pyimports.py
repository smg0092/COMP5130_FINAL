#ML imports 
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
from torch_geometric.nn import GATConv

#Custom imports
from Genetic_Functions_gnn import evaluate_member_fitness
from Memeber_gnn import Member

#Basic imports
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import time
import torch
import math

Citseer = Planetoid(root='data/Planetoid', name='Citeseer', transform=NormalizeFeatures())
Cora = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
PubMed = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())


