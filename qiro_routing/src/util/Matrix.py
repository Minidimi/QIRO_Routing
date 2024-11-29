from typing import Union
import numpy as np

class Matrix:
    """Defines a QUBO matrix that can be transformned into a Hamiltonian"""
    
    def __init__(self, size: int) -> None:
        """
        Parameters:
            size: Number of rows in the matrix
        """
        if size <= 0:
            raise ValueError("Matrix size should be a positive integer.")
        self.size = size
        self.matrix = np.zeros((size, size))

    def add_off_element(self, i: int, j: int, const: Union[int, float]) -> None:
        """
        Adds an off-diagonal element to the matrix.
        
        Parameters:
            i: Row index of the entry
            j: Column index of the entry
            const: Constant to be added
        """
        row, col = (np.abs(i), np.abs(j)) if np.abs(i) >= np.abs(j) else (np.abs(j), np.abs(i))
        self.matrix[row, col] += const

    def add_diag_element(self, i: int, const: Union[int, float]) -> None:
        """
        Adds a diagonal element to the matrix.
        
        Parameters:
            i: Index across the diagonal
            const: Constant to be added
        """
        self.matrix[np.abs(i), np.abs(i)] += const

    def get_entry(self, i: int, j: int):
        """
        Returns an entry of the matrix

        Parameters:
            i: Row index of the entry
            j: Column index of the entry 
        """
        row, col = (np.abs(i), np.abs(j)) if np.abs(i) >= np.abs(j) else (np.abs(j), np.abs(i))
        return self.matrix[row, col]