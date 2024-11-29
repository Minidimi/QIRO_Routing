import numpy as np
from qiskit_optimization import QuadraticProgram
import dimod

class Problem:
    """
    General problem generator base class.
    """

    def __init__(self, seed: int) -> None:
        """
        Parameters:
            seed: Seed for possible calculations
        """
        self.random_seed = seed
        self.type = 'default'
    
    def to_quadratic_program(self):
        """
        Transforms the QUBO matrix to a Qiskit QuadraticProgram
        """
        qp = QuadraticProgram()
        if self.type != 'CVRP':
            for i in range(self.matrix.shape[0] - 1):
                qp.binary_var()
            qp.minimize(quadratic=self.matrix[1:, 1:])
        else:
            for i in range(self.matrix.shape[0]):
                qp.binary_var()
            qp.minimize(quadratic=self.matrix)
        return qp
    
    def to_qiskit_hamilonian(self):
        """
        Transforms the QUBO matrix to a Qiskit Hamiltonian
        """
        qp = self.to_quadratic_program()
        qubit_op, _ = qp.to_ising()
        return qubit_op

    def to_dwave_bqm(self):
        """
        Transforms the QUBO matrix to a Dimod quadratic program compatible with D-Wave
        """
        qubo = self.to_quadratic_program()
        bqm_binary = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(), dimod.BINARY)
        return bqm_binary
    