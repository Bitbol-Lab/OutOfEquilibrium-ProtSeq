import copy
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import pinvh

import random
import math
import time
from typing import List, Tuple, Dict, Optional

from tqdm import tqdm


class MSAGenerator:
    """
    Class for generating Multiple Sequence Alignments (MSA) using Monte Carlo simulations.

    Attributes:
        num_nodes (int): Number of nodes in the system.
        num_spin_states (int): Number of possible states for each spin.
        field (np.ndarray): Local field values for the nodes.
        coupling (np.ndarray): Coupling values between nodes.
        temp_high (float): High temperature for random telegraph noise.
        temp_low (float): Low temperature for random telegraph noise.
        tau (float): Average duration of temperature state.
    """

    def __init__(
            self,
            field: np.ndarray,
            coupling: np.ndarray,
            temp_high: float = 1.0,
            temp_low: float = 1.0,
            tau: float = 100.0,
            max_mutations: int = 4000,
            switch_times: Optional[List[int]] = None,
    ) -> None:
        self.num_nodes = field.shape[0]
        self.num_spin_states = field.shape[1]
        self.field = field
        self.coupling = coupling
        self.temp_high = temp_high
        self.temp_low = temp_low
        self.tau = tau
        self.current_time = 0
        self.max_mutations = max_mutations
        random.seed(int(time.time()))

        # Pre-sample all temperature switches for the simulation
        self.switch_times = switch_times
        if self.switch_times is None:
            self.switch_times = MSAGenerator._pre_sample_switch_times(tau, max_mutations)
        self.switch_temps = MSAGenerator._pre_sample_switches_temps(self.switch_times, temp_high, temp_low)

    @staticmethod
    def _pre_sample_switch_times(tau, max_mutations) -> List[int]:
        """
        Pre-sample the switch .

        Returns:
            List[int]: Switch times.
        """
        switch_times = []
        current_time = 0

        while current_time < max_mutations:
            switch_times.append(current_time)
            current_time += int(-tau * math.log(random.random()))
        return switch_times

    @staticmethod
    def _pre_sample_switches_temps(switch_times, temp_high, temp_low) -> List[float]:
        """
        Get corresponding temperatures for the simulation.

        Returns:
            Tuple[List[float]]: Switch times and corresponding temperatures.
        """
        switch_temps = []
        current_temp = temp_low

        for _ in switch_times:
            switch_temps.append(current_temp)
            current_temp = temp_high if current_temp == temp_low else temp_low

        return switch_temps

    @staticmethod
    def _generate_chain_static(args: Tuple['MSAGenerator', int, int, Optional[List[int]], Optional[np.array]]) -> Tuple[int, List[Dict]]:
        """
        Static method to generate a single chain for parallel processing.

        Args:
            args (Tuple[MSAGenerator, int, int]): Tuple containing the generator instance, sequence index, and flips for equilibration.

        Returns:
            Tuple[int, List[Dict]]: Sequence index and state log.
        """
        instance, sequence_idx, num_flips_equilibration, keep_intermediate_states, initial_states = args
        return instance._generate_chain(sequence_idx, num_flips_equilibration, keep_intermediate_states, initial_states)

    def _generate_chain(self, sequence_idx: int,
                        num_flips_equilibration: int,
                        keep_intermediate_states: List[int],
                        initial_sequence: Optional[List[np.array]] = None) -> Tuple[int, List[np.array]]:
        """
        Generate a single chain using Monte Carlo sampling.

        Args:
            sequence_idx (int): Index of the sequence.
            num_flips_equilibration (int): Number of flips for equilibration.
            keep_intermediate_states (Optional[List[int]]): List of mutation indices to keep intermediate states.
            initial_states (Optional[List[np.array]]): Initial states for the chain.

        Returns:
            Tuple[int, List[np.array]]: Sequence index and state log.
        """
        msas: List[np.array] = []
        if initial_sequence is not None:
            assert len(initial_sequence) == self.num_nodes
            sequence = np.array(initial_sequence, dtype=np.int8)
        else:
            sequence = np.random.randint(0, self.num_spin_states, size=self.num_nodes, dtype=np.int8)
        self._mcmc(num_flips_equilibration, sequence, msas, keep_intermediate_states)
        return sequence_idx, msas

    def generate_msa_no_phylo(self, num_sequences: int,
                              num_flips_equilibration: int,
                              keep_intermediate_states: Optional[List[int]] = None,
                              n_workers: int = 4) -> np.ndarray:
        """
        Generate MSA using Monte Carlo sampling with consistent switches across sequences.

        Args:
            num_sequences (int): Number of sequences to generate.
            num_flips_equilibration (int): Flips for equilibration.
            keep_intermediate_states (Optional[List[int]]): List of mutation indices to keep intermediate states.
            n_workers (int): Number of parallel workers.

        Returns:
            np.ndarray: MSA and state logs.
        """
        msa = np.zeros((num_sequences, len(keep_intermediate_states), self.num_nodes), dtype=np.int8)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = tqdm(executor.map(
                self._generate_chain_static,
                [(self, idx, num_flips_equilibration, copy.deepcopy(keep_intermediate_states)) for idx in
                 range(num_sequences)],
            ), total=num_sequences, desc="Generating MSA")
            for sequence_idx, msas in results:
                for i, m in enumerate(msas):
                    msa[sequence_idx, i] = m

        return msa

    def generate_msa_phylo(self, initial_sequences: List[np.array],
                           num_flips_equilibration: int,
                           keep_intermediate_states: List[int],
                           n_workers: int = 4) -> np.ndarray:
        """
        Generate MSA using Monte Carlo sampling with consistent switches across sequences.

        Args:
            initial_sequences (List[np.array]): Initial sequences for the MSA.
            num_flips_equilibration (int): Flips for equilibration.
            keep_intermediate_states (Optional[List[int]]): List of mutation indices to keep intermediate states.
            n_workers (int): Number of parallel workers.

        Returns:
            np.ndarray: MSA and state logs.
        """
        num_sequences = len(initial_sequences)
        msa = np.zeros((num_sequences, len(keep_intermediate_states), self.num_nodes), dtype=np.int8)

        if n_workers == 1:
            for idx, seq in enumerate(initial_sequences):
                _, msas = self._generate_chain_static((self, idx, num_flips_equilibration, copy.deepcopy(keep_intermediate_states), seq))
                for i, m in enumerate(msas):
                    msa[idx, i] = m
            return msa

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = tqdm(executor.map(
                self._generate_chain_static,
                [(self, idx, num_flips_equilibration, copy.deepcopy(keep_intermediate_states), seq) for idx, seq in
                 enumerate(initial_sequences)],
            ), total=num_sequences, desc="Generating MSA")
            for sequence_idx, msas in results:
                for i, m in enumerate(msas):
                    msa[sequence_idx, i] = m

        return msa

    def _mcmc(self, num_mutations: int, spin_config: np.ndarray, msas: List[np.array],
              keep_intermediate_states: Optional[List[int]] = None) -> None:
        """
        Perform MCMC with pre-sampled temperature switches.

        Args:
            num_mutations (int): Total number of mutations.
            spin_config (np.ndarray): Spin configuration.
            msas (List[np.array]): List of generated MSAs.
            keep_intermediate_states (List[Dict]): List of mutation indices to keep intermediate states.
        """
        mutation_count = 0
        switch_idx = 0
        current_temp = self.switch_temps[switch_idx]

        while mutation_count < num_mutations:
            # Check if we need to switch temperature
            if switch_idx < len(self.switch_times) - 1 and mutation_count >= self.switch_times[switch_idx + 1]:
                switch_idx += 1
                current_temp = self.switch_temps[switch_idx]

            selected_node = random.randint(0, self.num_nodes - 1)
            new_state = random.randint(0, self.num_spin_states - 2)
            if new_state >= spin_config[selected_node]:
                new_state += 1

            new_hamiltonian = self._pseudo_hamiltonian(selected_node, new_state, spin_config)
            old_hamiltonian = self._pseudo_hamiltonian(selected_node, spin_config[selected_node], spin_config)
            delta_energy = new_hamiltonian - old_hamiltonian

            if delta_energy >= 0 or random.random() < math.exp(delta_energy / current_temp):
                spin_config[selected_node] = new_state
                mutation_count += 1
                if keep_intermediate_states is not None and mutation_count in keep_intermediate_states:
                    msas.append(spin_config.copy())
                """state_log.append({
                    "temperature": current_temp,
                    "mutation": mutation_count,
                    "sequence": spin_config.copy().tolist(),
                    "hamiltonian": new_hamiltonian,
                    "delta": delta_energy
                })"""

    def _pseudo_hamiltonian(self, node: int, node_state: int, spin_config: np.ndarray) -> float:
        """
        Compute the pseudo-Hamiltonian for a given node and state.

        Args:
            node (int): Node index.
            node_state (int): State of the node.
            spin_config (np.ndarray): Current spin configuration.

        Returns:
            float: Computed pseudo-Hamiltonian.
        """
        hamiltonian = self.field[node, node_state] - self.coupling[node, node, node_state, spin_config[node]]
        for neighbor_idx in range(spin_config.shape[0]):
            hamiltonian += self.coupling[node, neighbor_idx, node_state, spin_config[neighbor_idx]]
        return hamiltonian

    def segments_temperatures(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Calculate segments where the temperature remains constant based on pre-sampled switches.

        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
                Segments of high and low temperatures as (start, end) indices.
        """
        high_temperature_segments = []
        low_temperature_segments = []

        current_begin = 0
        current_temp = self.switch_temps[0]

        for i in range(1, len(self.switch_temps)):
            if self.switch_temps[i] != current_temp:
                # Record the segment
                if current_temp == self.temp_high:
                    high_temperature_segments.append((self.switch_times[current_begin], self.switch_times[i]))
                else:
                    low_temperature_segments.append((self.switch_times[current_begin], self.switch_times[i]))
                current_begin = i
                current_temp = self.switch_temps[i]

        # Add the final segment
        if current_temp == self.temp_high:
            high_temperature_segments.append((self.switch_times[current_begin], self.max_mutations))
        else:
            low_temperature_segments.append((self.switch_times[current_begin], self.max_mutations))

        return high_temperature_segments, low_temperature_segments


def apply_apc(matrix):
    """
    Apply Average Product Correction (APC) to a matrix.
    Args:
        matrix (np.ndarray): Input MI or direct coupling matrix.
    Returns:
        np.ndarray: APC-corrected matrix.
    """
    row_mean = matrix.mean(axis=1)
    col_mean = matrix.mean(axis=0)
    total_mean = matrix.mean()

    apc_matrix = np.outer(row_mean, col_mean) / total_mean
    return matrix - apc_matrix


def compute_dca(sequence_array, lambda_reg=0.01):
    """
    Compute the Direct Coupling Analysis (DCA) matrix.
    Args:
        sequence_array (np.ndarray): 2D array of encoded sequences (N_sequences x L_sequence).
        lambda_reg (float): Regularization parameter for covariance inversion.
    Returns:
        np.ndarray: Direct coupling matrix.
    """
    N, L = sequence_array.shape
    one_hot = np.zeros((N, L, 21))

    for i, seq in enumerate(sequence_array):
        one_hot[i, np.arange(L), seq] = 1

    one_hot_flat = one_hot.reshape(N, -1)
    cov_matrix = np.cov(one_hot_flat, rowvar=False)
    regularized_cov = cov_matrix + lambda_reg * np.eye(cov_matrix.shape[0])
    inv_cov = pinvh(regularized_cov)

    dca_matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            dca_matrix[i, j] = np.linalg.norm(
                inv_cov[i * 21:(i + 1) * 21, j * 21:(j + 1) * 21]
            )

    return dca_matrix


def mfDCA_with_contacts(sequence_array, lambda_reg=0.01):
    """
    Perform mfDCA and compute contact map using APC.
    Args:
        sequence_array (np.ndarray): 2D array of encoded sequences (N_sequences x L_sequence).
        lambda_reg (float): Regularization parameter for covariance inversion.
    Returns:
        np.ndarray: Contact map of size (L_sequence x L_sequence).
    """
    dca_matrix = compute_dca(sequence_array, lambda_reg=lambda_reg)
    apc_dca_matrix = apply_apc(dca_matrix)
    return apc_dca_matrix
