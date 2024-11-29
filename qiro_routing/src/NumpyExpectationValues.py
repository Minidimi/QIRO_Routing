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

class NumpyExpectationValues(ExpectationValues):  

    """
    Optimizes an exact solver with any number of layers and calculates correlations for QIRO
    """
    def __init__(self, problem: Problem, seed=100, logging=True):
        """
        Parameters:
            problem: Problem that should be optimized
            seed: Initial seed for optimization
            logging: Determines whether information should be displayed for logging
        """
        super().__init__(problem)

        backend = Aer.get_backend('aer_simulator_statevector')
        self.sampler=BackendSampler(backend=backend)

        self.logging = logging
        self.seed = seed

        self.operator = problem.to_qiskit_hamilonian()
        self.n_qubits = len(problem.matrix) - 1

        # Set the type of the expectation value
        self.type = "NumpyExpectationValue"

    def build_callback(self):
        """
        Builds callback function for optimizing QAOA
        """
        def callback_fn(count, params, val, best):
            self.params = params
        return callback_fn

    def calc_expect_val(self) -> (list, int, float):
        """
        Calculates all one- and two-point correlations and returns the one with highest absolute value
        """
        self.expect_val_dict = {}
        max_expect_val_location = -1
        max_expect_val = - np.inf
        max_expect_val_sign = np.sign(max_expect_val)

        for i in range(self.n_qubits):
            mixer_str = np.full(self.n_qubits, 'I', dtype=str)
            mixer_str[i] = 'Z'
            mixer = ''.join(mixer_str)
            mixer = SparsePauliOp([mixer])
            expect_val = self.state.expectation_value(mixer)

            self.expect_val_dict[frozenset({i})] = expect_val

            if expect_val > max_expect_val:
                max_expect_val = expect_val
                max_expect_val_location = i
                max_expect_val_sign = np.sign(expect_val)

        
        return max_expect_val_location, int(max_expect_val_sign), max_expect_val

    def optimize(self):
        """
        Optimizes the exact solver to minimize the energy.
        """
        solver = NumPyMinimumEigensolver()
        start_time = time.time()
        result = solver.compute_minimum_eigenvalue(self.operator)
        self.best_energy = result.eigenvalue
        self.state = result.eigenstate
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
        Returns the energy after the latest optimization
        """
        return self.best_energy
    
    def get_best_energy(self):
        """
        Uses exact solver to determine the optimal energy with regards to the Hamiltonian
        """
        return self.best_energy, sample_most_likely(self.state)

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
