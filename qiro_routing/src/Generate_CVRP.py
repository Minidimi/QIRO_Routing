from src.Generating_Problems import Problem
import copy
from src.util.Matrix import Matrix
import numpy as np
from typing import Union
import networkx as nx
from itertools import chain, product, combinations
from typing import Optional

class CVRP(Problem):
    """
    Defines the QUBO formulation of a CVRP instance with a problem-specific update step.
    """
    def __init__(self, graph: nx.Graph, vehicles: int, demands: Union[np.array, np.ndarray, list], capacity: int = 50, order: Optional[list] = None, alpha: float = 300, beta: float = 100, seed: int = 42) -> None:
        """
        Parameters:
            graph: Problem graph of the CVRP
            vehicles: Number of routes
            demands: Demands of each node
            capacity: Capacity of each vehicle; Here, each vehicle is considered to have the same capacity
            order: Previously fixed nodes from QIRO
            alpha: Penalty of the node constraints
            beta: Penalty of the capacity constraints
            seed: Random seed for calculations
        """
        super().__init__(seed)
        self.graph = copy.deepcopy(graph)
        self.order = order
        self.alpha = alpha
        self.vehicles = vehicles
        self.capacity = capacity
        self.n = len(self.graph.nodes)
        self.n_qubits = int(self.vehicles * self.n * self.n + self.vehicles * np.ceil(np.log2(self.capacity + 1)))  - self.n * sum([k != -1 for k in self.order])
        self.demands = demands
        self.beta = beta
        self.graph_to_matrix()
        self.position_translater = list(range(self.n_qubits))
        self.type = 'CVRP'

    def graph_to_matrix(self) -> None:
        """
        Transforms the problem graph into a QUBO matrix
        """
        self.matrixClass = Matrix(self.n_qubits)
        self.matrix = self.matrixClass.matrix
        self.calculate_objective_function()
        self.calculate_capacity_constraint()
        self.calculate_other_constraints()


    def calculate_capacity_constraint(self):
        """
        Determines the terms for the capacity constraint and adds them to the matrix
        """
        for k in range(self.vehicles):
            c3_single = np.zeros(self.n_qubits)
            c3_single_c = np.zeros(self.n_qubits)
            t_shift = 0
            k_shift = k * self.n * self.n + self.get_shift(k)
            for t in range(0, self.n):
                if self.order[self.n * k + t] != -1:
                    t_shift -= 1
                    continue
                for v in range(1, self.n):
                    c3_single[k_shift  + (t + t_shift) * self.n + v] = self.demands[v]
            for b in range(int(np.ceil(np.log2(self.capacity + 1)))):
                ind = int(self.vehicles * self.n * self.n + self.get_shift(self.vehicles) + k * np.ceil(np.log2(self.capacity + 1)) + b)
                c3_single[ind] = 2**b

            remaining_capacity = self.capacity
            for i in range(len(self.graph.nodes)):
                node = int(self.order[k * self.n + i])
                if node > 0:
                    remaining_capacity -= self.demands[node]
            c3_single_c = c3_single * -2 * remaining_capacity

            for i in range(len(c3_single)):
                self.matrixClass.add_diag_element(i, self.beta * c3_single_c[i])
                for j in range(len(c3_single)):
                    self.matrixClass.add_off_element(i, j, self.beta * c3_single[i] * c3_single[j])
    
    def calculate_objective_function(self):
        """
        Determines the terms for the CVRP distance and adds them to the matrix
        """
        weight_dict = nx.get_edge_attributes(self.graph, 'weight')
        for k in range(self.vehicles):
            t_shift = 0
            k_shift = k * self.n * self.n + self.get_shift(k)
            for t in range(self.n):
                if self.order[k * self.n + t] < 0 and self.order[k * self.n + (t + 1) % self.n] < 0:
                    for i in range(self.n):
                        for j in range(self.n):
                            if i != j:
                                edge = (i, j) if i < j else (j, i)
                                k_shift = k * self.n * self.n + self.get_shift(k)
                                self.matrixClass.add_off_element(k_shift + ((t + t_shift) % (self.get_steps_set(k))) * self.n + i, k_shift + ((t + t_shift + 1) % (self.get_steps_set(k))) * self.n + j, weight_dict[edge])
                elif self.order[k * self.n + t] < 0 and self.order[k * self.n + (t + 1) % self.n] >= 0:
                    for i in range(self.n):
                        edge = (self.order[k * self.n + (t + 1) % self.n], i) if i >= self.order[k * self.n + (t + 1) % self.n] else (i, self.order[k * self.n + (t + 1) % self.n])
                        k_shift = k * self.n * self.n + self.get_shift(k)
                        if edge in weight_dict:
                            self.matrixClass.add_diag_element(k_shift + ((t + t_shift) % (self.get_steps_set(k))) * self.n + i, weight_dict[edge])
                elif self.order[k * self.n + t] >= 0 and self.order[k * self.n + (t + 1) % self.n] < 0:
                    t_shift -= 1
                    for i in range(self.n):
                        edge = (self.order[k * self.n + t], i) if i >= self.order[k * self.n + t] else (i, self.order[k * self.n + t])
                        k_shift = k * self.n * self.n + self.get_shift(k)
                        if edge in weight_dict:
                            self.matrixClass.add_diag_element(k_shift + ((t + t_shift + 1) % (self.get_steps_set(k))) * self.n + i, weight_dict[edge])
                else:
                    t_shift -= 1

    def calculate_other_constraints(self):
        """
        Determines the terms for the node constraint and adds them to the matrix
        """
        for i in range(1, self.n):
            for k1 in range(self.vehicles):
                k1_shift = k1 * self.n * self.n + self.get_shift(k1)
                for t1 in range(self.get_steps_set(k1)):
                    if i in self.order:
                        self.matrixClass.add_diag_element(k1_shift + ((t1) % (self.get_steps_set(k1))) * self.n + i, self.alpha)
                    else:
                        self.matrixClass.add_diag_element(k1_shift + ((t1) % (self.get_steps_set(k1))) * self.n + i, -2 * self.alpha)
                    for k2 in range(self.vehicles):
                        k2_shift = k2 * self.n * self.n + self.get_shift(k2)
                        for t2 in range(self.get_steps_set(k2)):
                            self.matrixClass.add_off_element(k1_shift + ((t1) % (self.get_steps_set(k1))) * self.n + i, k2_shift + ((t2) % (self.get_steps_set(k2))) * self.n + i, self.alpha)
        
        for k in range(self.vehicles):
            k_shift = k * self.n * self.n + self.get_shift(k)
            for t in range(self.get_steps_set(k)):
                for i in range(self.n):
                    self.matrixClass.add_diag_element(k_shift + ((t) % (self.get_steps_set(k))) * self.n + i, -2 * self.alpha)
                    for j in range(self.n):
                        self.matrixClass.add_off_element(k_shift + ((t) % (self.get_steps_set(k))) * self.n + i, k_shift + ((t) % (self.get_steps_set(k))) * self.n + j, self.alpha)
                    

    def get_shift(self, k):
        """
        Helper function for indexing
        """
        reduced_order = self.order[:self.n * k]
        number_set = sum([k != -1 for k in reduced_order])
        return - number_set * self.n
    
    def get_steps_set(self, k):
        """
        Helper function for indexing
        """
        reduced_order = self.order[k * self.n : (k + 1) * self.n]
        number_set = sum([k != -1 for k in reduced_order])
        return self.n - number_set
    
    def get_k_order(self, k):
        """
        Helper function for indexing
        """
        return self.order[self.n * k : self.n * (k + 1)]
