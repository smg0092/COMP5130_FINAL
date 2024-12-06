import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from Memeber_gnn import Member

#Basic imports
import random as rd
import numpy as np
import torch


Citseer = Planetoid(root='data/Planetoid', name='Citeseer', transform=NormalizeFeatures())
Cora = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
PubMed = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

def evaluate_member_fitness(member, dataset, num_epochs):
    """
    Evaluate the accuracy of the graph represented by the given Member object.
    :param member: A Member object representing a graph with a chromosome.
    :param model: A trained PyTorch Geometric GNN model.
    :param data: The PyTorch Geometric Data object corresponding to the original graph.
    :return: Accuracy of the model on the reconstructed graph.
    """

    data = member.data
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            # torch.manual_seed(1234567)
            self.conv1 = GCNConv(dataset.num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    model = GCN(hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Example: Train the model (simplified)
    for epoch in range(1, num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Reconstruct the graph using the Member's chromosome
    reconstructed_edge_index = member.reconstruct_graph()

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Perform a forward pass
        out = model(data.x, reconstructed_edge_index)
        predictions = out.argmax(dim=1)  # Get predicted classes

    # Calculate accuracy on the test set
    correct = (predictions[data.test_mask] == data.y[data.test_mask]).sum()
    accuracy = int(correct) / int(data.test_mask.sum())

    max_weight = len(member.chromosome)
    the_weight = member.weight

    fitness = round(((1 - accuracy) * 100.00), 2) + ((1.0/((the_weight/max_weight) + 1.01))*0.0001)
    return fitness, accuracy

def generate_population(n_population: int, graph):
    """
    - Fills a list with a member objects and gives each of them chromosomes the size of n_features 
    """
    population = []

    for i in range(n_population):
        member = Member(graph)

        #The chromosome of a member getting changed
        member.mutate()
        population.append(member)

    return population

def evaluate_population_gnn(population: list):
    """
    - The current evaluation function for GEFES. It uses a perceptron and evaluates each memeber of the population with a corresponding fitness
    """

    for i in range(len(population)):
        fitness, accuracy = evaluate_member_fitness(population[i], population[i].data, 101)
        population[i].add_fitness(fitness)
        population[i].add_accuracy(accuracy)

        
def create_offspring(population: list, data):
    """
    - Takes to random members of the population and uses uniform crossover and bit-mutation to form their child
    """

    mom_index = rd.randint(0, len(population)-1)
    dad_index = rd.randint(0, len(population)-1)

    mom = population[mom_index]
    dad = population[dad_index]


    offspring = Member(data)

    #uniform crossover
    for i in range(len(mom.chromosome)):
        random_point = rd.uniform(0,1)

        if (random_point > 0.5): 
            offspring.chromosome[i] = mom.chromosome[i]
        else:
            offspring.chromosome[i] = dad.chromosome[i]

    #Random Bit mutation
    for i in range(len(mom.chromosome)):
        random_point = rd.uniform(0,1)

        if (random_point < 0.1):

            if(offspring.chromosome[i] == 0):

                offspring.chromosome[i] = 1
            else:
                offspring.chromosome[i] = 0 

    return offspring

def evaluate_offspring_basic(offspring):
    """
    - Gets the fitness of the new offspring
    """
    fitness, accuracy = evaluate_member_fitness(offspring, offspring.data, 101)
    offspring.add_fitness(fitness)
    offspring.add_accuracy(accuracy)

def replace_worstFit(population: list, offspring: object):
    """
    - Finds the member with the worst fitness using a minimum search and replaces them with the offspring
    """
    min_val = population[0].fitness
    worst_member_index = 0

    for i in range(len(population)):
        if population[i].fitness < min_val:
            min_val = population[i].fitness
            worst_member_index = i

    population[worst_member_index] = offspring        

def Genetic_Algo(n_population: int, graph, iterations: int):
    chromosome_list = []

    """
    -----------------------------------------
    - Genetic Algorithm which uses the data itself as the features, not patches
    -----------------------------------------
    """
    population = generate_population(n_population, graph)
    evaluate_population_gnn(population)

    with open(f"Gene_Algo_GNN.txt", "w") as file:

        for i in range(0, iterations):
            file.write(f"Iteration: {i+1}\n")
            
            for j in range(len(population)):
                file.write(f"{population[j].fitness} {population[j].accuracy}\n")
                chromosome_list.append(population[j].chromosome)
                file.flush()

            file.write("\n")

            offspring = create_offspring(population, graph)

            evaluate_offspring_basic(offspring) 
            replace_worstFit(population, offspring) 

    chromosome_list = np.array(chromosome_list)

    np.save(arr=chromosome_list, file="chromosome_list")


dataset = Citseer

# print('======================')
# print(f'Dataset: {dataset}')
# print(f'Data type:')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')
# print('======================')

# data = dataset[0]  # Get the first graph object.
# print(len(data))
# print("Data type of data graph object: ")
# print(type(data))

# # member = Member(data)
# # Evaluate_gnn.evaluate_member_fitness(member, member.data,101)
# start = time.time()
# Genetic_Algo(graph=data, dataset= dataset, n_population=100, iterations=10)
# end = time.time()

# elapsed_time = (end - start) / 60
# print(f"This took {elapsed_time:.2f} minutes")
