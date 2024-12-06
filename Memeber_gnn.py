#population = [memeber, member, member]
import torch
import random as rd 

class Member:
    def __init__(self, data):
        """
        Initialize a Member object from a PyTorch Geometric Data object.
        :param data: A PyTorch Geometric Data object.
        """
        self.data = data
        self.edge_index = data.edge_index  # Tensor of shape (2, num_edges)
        self.num_edges = self.edge_index.shape[1]
        self.chromosome = self.generate_chromosome()
        self.weight = Member.get_wght(self, self.chromosome)
        self.accuracy = 0
        self.fitness = 0

    def generate_chromosome(self):
        """
        Generate a chromosome from the edge_index.
        Each edge is represented as a bit (1 or 0).
        """
        return [1] * self.num_edges  # Start with all edges present

    def reconstruct_graph(self):
        """
        Reconstruct the edge_index tensor from the chromosome.
        Returns:
            A new edge_index tensor where edges marked as 0 are removed.
        """
        mask = torch.tensor(self.chromosome, dtype=torch.bool)
        return self.edge_index[:, mask]

    def reconstruct_graph_hitlist(self, hitlist):
        """
        Reconstruct the edge_index tensor from the chromosome, considering only edges involving nodes in the hitlist.
        :param hitlist: List of nodes whose edges are subject to reconstruction.
        :return: A new edge_index tensor with edges marked as 0 removed for hitlist nodes.
        """
        # Create a mask to filter edges in edge_index based on the hitlist
        hitlist_mask = torch.zeros(self.edge_index.shape[1], dtype=torch.bool)

        # Find edges involving nodes in the hitlist
        for node in hitlist:
            # Check both source (row 0) and destination (row 1) for matches
            hitlist_mask |= (self.edge_index[0] == node) | (self.edge_index[1] == node)

        # Combine the hitlist mask with the chromosome mask
        chromosome_mask = torch.tensor(self.chromosome, dtype=torch.bool)
        final_mask = chromosome_mask & hitlist_mask  # Keep only hitlist edges affected by the chromosome

        # Apply the final mask to edge_index
        return self.edge_index[:, final_mask]

    def remove_edges(self, num_edges_to_remove=1000):
        """
        Remove a specified number of edges by setting their corresponding bits to 0 in the chromosome.
        :param num_edges_to_remove: Number of edges to remove.
        """
        # Get the indices of edges currently set to 1 (existing edges)
        edge_indices = [i for i, bit in enumerate(self.chromosome) if bit == 1]

        # Ensure we do not attempt to remove more edges than are present
        edges_to_remove = min(num_edges_to_remove, len(edge_indices))

        # Randomly select edges to remove
        edges_to_remove_indices = rd.sample(edge_indices, edges_to_remove)

        # Set the selected edges to 0 in the chromosome
        for index in edges_to_remove_indices:
            self.chromosome[index] = 0
    
    def mutate(self, mutation_rate=0.1):
        """
        Mutate the chromosome by flipping random bits.
        """
        for i in range(len(self.chromosome)):
            if rd.random() < mutation_rate:
                self.chromosome[i] = 1 - self.chromosome[i]  # Flip bit

    def get_wght(self, chromosome) -> int:
        weight = 0

        for i in range(len(chromosome)):
            weight += chromosome[i]

        return weight
    
    def add_fitness(self, fit_to_add):
        """
        Update the fitness score for the member.
        """
        self.fitness = fit_to_add


    def add_accuracy(self, acc_in):
        self.accuracy = acc_in


    def display_fitness(self):
        print(f"Fitness: {self.fitness}")

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data


# Define a small graph with 4 nodes and 4 edges
def test_small_Graph():
    edge_index = torch.tensor([[0, 1, 2, 3],
                            [1, 2, 3, 0]], dtype=torch.long)  # Directed edges: (0->1), (1->2), (2->3), (3->0)

    # Define node features (optional, set as a placeholder)
    x = torch.tensor([[1], [1], [1], [1]], dtype=torch.float)

    # Create the graph as a PyTorch Geometric Data object
    small_graph = Data(x=x, edge_index=edge_index)

    # Print the graph for verification
    print("Small Graph:")
    print(small_graph)

    # Create a Member object
    member = Member(small_graph)
    print(member.edge_index)

    member.chromosome = [0,1,1,0]

    new_edge_index = member.reconstruct_graph()

    print(new_edge_index)


