import copy
import numpy as np
import itertools as it

class QIRO:
    """
    General QIRO base class. Here only used as a wrapper for the quantum algorithm and the problem
    """
    def __init__(self, nc, expectation_values):
        self.problem = expectation_values.problem
        self.nc = nc
        self.expectation_values = expectation_values
        self.assignment = []
        self.solution = []
