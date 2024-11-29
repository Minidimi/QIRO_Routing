from src.ExpVal import ExpectationValues
from src.Generating_Problems import Problem
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit import BasicAer
from qiskit.primitives import Sampler, BackendSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.optimizers import SPSA, COBYLA, NFT, UMDA
from qiskit.circuit.library import QAOAAnsatz
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

class MultiLayerQAOAExpectationValues(ExpectationValues):  
    """
    Optimizes QAOA with any number of layers and calculates correlations for QIRO
    """
    def __init__(self, problem: Problem, reps: int = 1, seed: int = 100, logging: bool = True, correlations_best: bool = False):
        """
        Parameters:
            problem: Problem that should be optimized
            reps: Number of QAOA layers
            seed: Initial seed for optimization
            logging: Determines whether information should be displayed for logging
            correlations_best: Determines whether only the best samples should be used when computing the correlations; If False, all samples are used
        """
        super().__init__(problem)

        backend = Aer.get_backend('aer_simulator_statevector')
        self.sampler=BackendSampler(backend=backend)

        self.logging = logging
        self.seed = seed

        self.reps = reps
        self.operator = problem.to_qiskit_hamilonian()
        self.n_qubits = len(problem.matrix) - 1

        #Initialize mixer Hamiltonian
        mixer_str = np.full((self.n_qubits, self.n_qubits), 'I', dtype=str)
        np.fill_diagonal(mixer_str, 'X')
        mixer_str = [''.join(row) for row in mixer_str]
        self.mixer = SparsePauliOp(mixer_str)

        # Initialize ansatz
        self.ansatz = QAOAAnsatz(cost_operator=self.operator, reps=reps, mixer_operator=self.mixer)

        self.params = np.zeros(reps * 2)

        self.type = "MultiLayerQAOAExpectationValue"
        self.correlations_best = correlations_best

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
        ansatz = self.ansatz.assign_parameters(self.params)
        if self.correlations_best:
            self.state_dict = self.get_best_states()
        max_expect_val_location = -1
        max_expect_val = - np.inf
        max_expect_val_sign = np.sign(max_expect_val)

        for i in range(self.n_qubits):
            mixer_str = np.full(self.n_qubits, 'I', dtype=str)
            mixer_str[i] = 'Z'
            mixer = ''.join(mixer_str)
            mixer = SparsePauliOp([mixer])
            expect_val = 0
            for key in self.state_dict:
                if key[i] == '0':
                    expect_val += self.state_dict[key]
                else:
                    expect_val -= self.state_dict[key]


            self.expect_val_dict[frozenset({self.n_qubits - i - 1})] = expect_val

            if expect_val > max_expect_val:
                max_expect_val = expect_val
                max_expect_val_location = i
                max_expect_val_sign = np.sign(expect_val)

        
        return max_expect_val_location, int(max_expect_val_sign), max_expect_val

    def optimize(self):
        """
        Optimizes QAOA parameters to minimize the energy.
        """
        n_params = self.ansatz.num_parameters
        np.random.seed(self.seed)
        init_params = np.random.random(n_params) * 2 * np.pi


        qaoa = QAOA(self.sampler, NFT(maxiter=1000), reps=self.reps, mixer=self.mixer, callback=self.build_callback(), initial_point=init_params)
        start_time = time.time()
        result = qaoa.compute_minimum_eigenvalue(self.operator)
        self.best_energy = result.eigenvalue
        self.state_dict = result.eigenstate.binary_probabilities(self.n_qubits)
        self.state = self.get_best_state()
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
        ansatz = self.ansatz.assign_parameters(self.params)
        state = Statevector(bound_circuit_to_instruction(ansatz))
        return state.expectation_value(self.operator).real
    
    def get_best_energy(self):
        """
        Uses exact solver to determine the optimal energy with regards to the Hamiltonian
        """
        qubit_op = self.operator
        solver = NumPyMinimumEigensolver()
        start_time = time.time()
        result = solver.compute_minimum_eigenvalue(qubit_op)
        end_time = time.time()
        if self.logging:
            print('Calculating best energy took', end_time - start_time, 'seconds')
        return result.eigenvalue, sample_most_likely(result.eigenstate)

    def get_most_likely_state(self, dist: dict):
        """
        Returns the state sampled the most
        """
        probs = dist.binary_probabilities(num_bits=self.n_qubits)
        state = max(probs, key=probs.get)
        state_vec = [int(i) for i in state]
        state_vec.reverse()
        return state_vec
    
    def get_best_states(self):
        """
        Returns a dict of the states among the samples with the best energy
        """
        min_energy = np.inf
        for state in self.state_dict:
            state_vec = [int(i) for i in state]
            state_vec.reverse()
            state_vec = [0] + state_vec
            state_vec = np.array(state_vec)
            energy = state_vec.T @ self.problem.matrix @ state_vec
            if energy < min_energy:
                min_energy = energy
                best_state = state_vec[1:]
        
        best_states = {}
        best_states_prob = 0
        for state in self.state_dict:
            state_vec = [int(i) for i in state]
            state_vec.reverse()
            state_vec = [0] + state_vec
            state_vec = np.array(state_vec)
            energy = state_vec.T @ self.problem.matrix @ state_vec
            if energy <= min_energy:
                best_states[state] = self.state_dict[state]
                best_states_prob += self.state_dict[state]
        
        for key in best_states:
            best_states[key] = best_states[key] / best_states_prob
        return best_states
    
    def get_best_state(self):
        """
        Returns a singular best state among the samples
        """
        min_energy = np.inf
        for state in self.state_dict:
            state_vec = [int(i) for i in state]
            state_vec.reverse()
            state_vec = [0] + state_vec
            state_vec = np.array(state_vec)
            energy = state_vec.T @ self.problem.matrix @ state_vec
            if energy < min_energy:
                min_energy = energy
                best_state = state_vec[1:]
        return best_state


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
