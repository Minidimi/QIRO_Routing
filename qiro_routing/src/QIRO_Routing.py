from src.QIRO import QIRO
import copy
import networkx as nx
import numpy as np
from typing import Union

class QIRO_TSP(QIRO):
    """
    QIRO wrapper for the TSP
    """
    def __init__(self, nc, expectation_values):
        super().__init__(nc, expectation_values)
        self.graph = copy.deepcopy(self.problem.graph)

class QIRO_CVRP(QIRO):
    """
    QIRO wrapper for the CVRP
    """
    def __init__(self, nc, expectation_values):
        super().__init__(nc, expectation_values)
        self.graph = copy.deepcopy(self.problem.graph)

    