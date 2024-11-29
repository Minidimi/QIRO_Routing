from src.ExpVal import ExpectationValues
from src.Generating_Problems import Problem
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import QAOA, SamplingVQE
from qiskit import BasicAer
from qiskit.primitives import Sampler, BackendSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.optimizers import SPSA, COBYLA, NFT, UMDA
from qiskit.circuit.library import QAOAAnsatz, TwoLocal
from qiskit.primitives.utils import (
    _circuit_key,
    _observable_key,
    bound_circuit_to_instruction,
    init_observable,
)
from qiskit.quantum_info import Statevector
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
import time
from qiskit import Aer
from neal import SimulatedAnnealingSampler

class SAExpectationValues(ExpectationValues):  
    """
    Optimizes simulated annealing with any number of layers and calculates correlations for QIRO
    """
    def __init__(self, problem: Problem, num_reads=1000, seed=100, logging=True, correlations_best=False):
        """
        Parameters:
            problem: Problem that should be optimized
            num_reads: Number of samples
            seed: Initial seed for optimization
            logging: Determines whether information should be displayed for logging
            correlations_best: Determines whether only the best samples should be used when computing the correlations; If False, all samples are used
        """
        super().__init__(problem)

        backend = Aer.get_backend('aer_simulator_statevector')
        self.sampler=BackendSampler(backend=backend)

        self.logging = logging
        self.n_qubits = len(problem.matrix) - 1
        self.num_reads = num_reads

        self.type = "SAExpectationValue"
        self.correlations_best=correlations_best
        np.random.seed(seed)

    def calc_expect_val(self) -> (list, int, float):
        """
        Calculates all one- and two-point correlations and returns the one with highest absolute value
        """
        self.expect_val_dict = {}
        max_expect_val_location = -1
        max_expect_val = - np.inf
        max_expect_val_sign = np.sign(max_expect_val)

        for i in range(self.n_qubits):
            num_entries = 0
            total = 0
            for entry in self.sampleset:
                num_entries += 1
                if entry[i] == 1:
                    total += -1
                else:
                    total += 1
            expect_val = total / num_entries

            self.expect_val_dict[frozenset({i})] = expect_val

            if expect_val > max_expect_val:
                max_expect_val = expect_val
                max_expect_val_location = i
                max_expect_val_sign = np.sign(expect_val)

        
        return max_expect_val_location, int(max_expect_val_sign), max_expect_val

    def optimize(self):
        """
        Optimizes simulated annealing to minimize the energy.
        """
        start_time = time.time()
        sampler = SimulatedAnnealingSampler()
        if self.correlations_best:
            self.sampleset = sampler.sample(self.problem.to_dwave_bqm(), num_reads=self.num_reads).lowest()
        else:
            self.sampleset = sampler.sample(self.problem.to_dwave_bqm(), num_reads=self.num_reads)
        self.state = self.sampleset.first.sample.values()
        self.best_energy = self.sampleset.first.energy
        end_time = time.time()
        if self.logging:
            print('Optimization took', end_time - start_time, 'seconds')

        start_time = time.time()
        # computing the correlatios at the optimal parameters
        (
            max_expect_val_location,
            max_expect_val_sign,
            max_expect_val,
        ) = self.calc_expect_val()
        end_time = time.time()
        if self.logging:
            print('Calculating correlations took', end_time - start_time, 'seconds')
        self.fixed_correl.append(
            [max_expect_val_location, max_expect_val_sign, max_expect_val]
        )

        return max_expect_val_location, max_expect_val_sign, max_expect_val
    
    def get_energy(self):
        """
        Returns the energy for the current configuration of parameters
        """
        return self.best_energy
    
    def get_best_energy(self):
        return self.best_energy, self.state

def bitfield(n, L):
    """
    Helper function to transform state into binary representation
    """
    result = np.binary_repr(n, L)
    return [int(digit) for digit in result]

def sample_most_likely(state_vector):
    """
    Helper function for exact solver to sample the result
    """
    values = state_vector
    n = int(np.log2(len(values)))
    k = np.argmax(np.abs(values))
    x = bitfield(k, n)
    x.reverse()
    return np.asarray(x)
