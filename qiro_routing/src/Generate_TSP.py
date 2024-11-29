from src.Generating_Problems import Problem
import copy
from src.util.Matrix import Matrix
import numpy as np
from typing import Union
import networkx as nx
from itertools import chain, product, combinations
from typing import Optional


class TSP(Problem):
    """
    Defines the QUBO formulation of a TSP instance with a problem-specific update step.
    """

    def __init__(self, graph: nx.Graph, order: Optional[list] = None, alpha: float = 200, seed: int = 42) -> None:
        """
        Parameters:
            graph: Problem graph of the TSP
            order: Previously fixed nodes from QIRO
            alpha: Penalty of the feasibility constraints
            seed: Random seed for calculations
        """
        super().__init__(seed=seed)
        self.graph = copy.deepcopy(graph)
        if order is None:
            order = [-1] * len(self.graph.nodes)
        self.order = copy.deepcopy(order)
        self.alpha = alpha
        self.var_list = None
        self.position_translater = None
        self.type = 'TSP'

        # compute the matrix (i.e., Hamiltonian) from the graph. Also sets the varlist!
        self.graph_to_matrix()
        self.remain_var_list = copy.deepcopy(self.var_list)

    def graph_to_matrix(self) -> None:
        """
        Transforms the problem into a QUBO matrix
        """
        weight_dict = nx.get_edge_attributes(self.graph, 'weight')
        n = sum([k < 0 for k in self.order])
        num_nodes = len(self.graph.nodes)

        self.matrixClass = Matrix(n * n + 1)
        self.matrix = self.matrixClass.matrix

        #Prepare indexing depending on previous QIRO steps
        node_values = {}
        i = 0
        for i0 in range(num_nodes):
            if i0 not in self.order:
                node_values[i0] = i
                i += 1
        
        step_values = {}
        i = 0
        for t in range(num_nodes):
            if self.order[t] < 0:
                step_values[t] = i
                i += 1
        #Calculate TSP distance cost
        n_set = 0
        for t in range(num_nodes):
            if self.order[t] >= 0 and self.order[(t + 1) % num_nodes] < 0:
                n_set += 1
                for i in range(num_nodes):
                    if i not in self.order:
                        edge = (self.order[t], i) if i >= self.order[t] else (i, self.order[t])
                        self.matrixClass.add_diag_element(get_ind(node_values[i], step_values[(t + 1) % num_nodes], n) + 1, weight_dict[edge])
            elif self.order[t] < 0 and self.order[(t + 1) % num_nodes] >= 0:
                for i in range(num_nodes):
                    if i not in self.order:
                        edge = (self.order[(t + 1) % num_nodes], i) if i >= self.order[(t + 1) % num_nodes] else (i, self.order[(t + 1) % num_nodes])
                        self.matrixClass.add_diag_element(get_ind(node_values[i], step_values[t], n) + 1, weight_dict[edge])
            elif self.order[t] < 0 and self.order[(t + 1) % num_nodes] < 0:
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i in self.order or j in self.order or i == j:
                            continue
                        edge = (i, j) if i < j else (j, i)
                        self.matrixClass.add_off_element(get_ind(node_values[i], step_values[t], n) + 1, get_ind(node_values[j], step_values[(t + 1) % num_nodes], n) + 1, weight_dict[edge])

        #Calculate constraints
        for t in range(n):
            for i1 in range(n):
                self.matrixClass.add_diag_element(get_ind(i1, t, n) + 1, -2 * self.alpha)
                for i2 in range(n):
                    self.matrixClass.add_off_element(get_ind(i1, t, n) + 1, get_ind(i2, t, n) + 1, self.alpha)

        for i in range(n):
            for t1 in range(n):
                self.matrixClass.add_diag_element(get_ind(i, t1, n) + 1, -2 * self.alpha)
                for t2 in range(n):
                   self.matrixClass.add_off_element(get_ind(i, t1, n) + 1, get_ind(i, t2, n) + 1, self.alpha)

        self.position_translater = list(range(n * n))


class QubitTSP(TSP):
    """
        Defines the QUBO matrix of a TSP instance with a qubit-level update step
    """
    def __init__(self, graph: nx.Graph, order: Optional[list] = None, alpha: float = 200, seed: int = 42, fixed: list = []) -> None:
        """
        Parameters:
            graph: Problem graph of the TSP
            order: Previously fixed nodes from QIRO
            alpha: Penalty of the feasibility constraints
            seed: Random seed for calculations
        """
        self.fixed = fixed
        self.index_mapper = {}
        super().__init__(graph, order, alpha, seed)
        
    
    def graph_to_matrix(self) -> None:
        #First, generate normal matrix for the initial configuration
        super().graph_to_matrix()
        n = sum([k < 0 for k in self.order])
        num_nodes = len(self.graph.nodes)

        #Add linear terms if one entry of a diagonal term has been set to 1
        new_matrix_class = Matrix((n * n + 1) - sum(self.fixed != -1))
        for i in range(1, len(self.matrix)):
            for j in range(len(self.fixed)):
                if self.fixed[j] == 1:
                    self.matrixClass.add_diag_element(i, self.matrix[i, j + 1])
                    self.matrixClass.add_diag_element(i, self.matrix[j + 1, i])
        
        i_shift = 0
        for i in range(1, len(self.matrix)):
            j_shift = 0
            if self.fixed[i - 1] != -1:
                i_shift -= 1
                skip = True
            for j in range(1, len(self.matrix)):
                skip = False
                if self.fixed[j - 1] != -1:
                    j_shift -= 1
                    skip = True
                if not skip:
                    self.index_mapper[i + i_shift - 1] = i - 1
                    self.index_mapper[j + j_shift - 1] = j - 1
        
        for i in reversed(range(len(self.fixed))):
            if self.fixed[i] != -1:
                self.matrix = np.delete(self.matrix, i + 1, 0)
                self.matrix = np.delete(self.matrix, i + 1, 1)
        new_matrix_class.matrix = self.matrix

        self.matrixClass = new_matrix_class
        self.matrix = new_matrix_class.matrix
        #print(self.matrix)



# Misc. helper functions

def get_ind(i, t, n):
        i = i % n
        t = t % n
        return t * n + i


