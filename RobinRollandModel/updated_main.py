"""
Module for equilibrium charge distribution and evaporation simulation based on the RR model.
"""

import os
import logging
import numpy as np
import h5io
from tqdm import tqdm
from scipy.constants import physical_constants
from .datautils import TipGenerator

# Configure logging
logging.basicConfig(
    filename="simulation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class SimulationConfig:
    """
    Configuration class for simulation parameters.
    """

    def __init__(
        self,
        e_field=4.0,
        radius=20.0,
        steps=1000,
        epsilon=1e-9,
        zheight=50.0,
        nn_n=12,
        relaxed=False,
        nd=3.0,
    ):
        self.e_field = e_field
        self.radius = radius
        self.steps = steps
        self.epsilon = epsilon
        self.zheight = zheight
        self.nn_n = nn_n
        self.relaxed = relaxed
        self.nd = nd


class RRModel:
    """
    A class representing the RRModel for simulating charge distribution,
    evaporation trajectory, and other related physical processes on a
    nanostructure's surface under an external electric field.

    Example usage:
    --------------
    >>> from rrmodel import RRModel
    >>> from datautils import TipGenerator
    >>> tip_gen = TipGenerator(...)
    >>> structure = tip_gen.create_tip()
    >>> model = RRModel(tip_gen, structure, e_field=5.0)
    >>> model.run_evaporation(num_atoms=10)
    """

    def __init__(self, tip_generator, structure, e_field):
        """
        Initializes the RRModel instance.

        Parameters
        ----------
        tip_generator : TipGenerator
            An instance of TipGenerator class.
        structure : Atoms
            An object representing the nanostructure.
        e_field : float
            The external electric field magnitude.
        """
        if not isinstance(tip_generator, TipGenerator):
            raise TypeError("tip_generator must be an instance of TipGenerator class.")
        if not hasattr(structure, "positions"):
            raise TypeError("structure must have a 'positions' attribute.")
        if not isinstance(e_field, (int, float)):
            raise TypeError("e_field must be a numeric value.")

        self.tip_generator = tip_generator
        self.structure = structure
        self.e_field = e_field

        # Constants
        self.a_to_bohr = (
            1 / physical_constants["Bohr radius"][0] * 1e10
        )  # Angstrom to Bohr
        self.eV_to_Hartree = physical_constants["electron volt-hartree relationship"][0]
        self.right_field_conversion = 1 / 51.4  # Conversion factor for electric field

        # Initialize state variables
        self.charge_distribution = None
        self.surface_indices = None
        self.surface_normals = None

    @staticmethod
    def nunique(arr):
        """
        Counts the unique elements in the given array.

        Parameters
        ----------
        arr : numpy.ndarray
            The array for which unique elements and their counts are to be found.

        Returns
        -------
        int
            The count of the first unique element found in `arr`.
        """
        _, counts = np.unique(arr, return_counts=True)
        return counts[0]

    @staticmethod
    def unitnormals(nn_vectors, nn_shells):
        """
        Computes unit normal vectors for each reference point based
        on nearest neighbors and shell information.

        Parameters
        ----------
        nn_vectors : numpy.ndarray
            The nearest neighbor vectors for each reference point.
        nn_shells : numpy.ndarray
            The shell indices for nearest neighbors.

        Returns
        -------
        numpy.ndarray
            The unit normal vectors for each reference point.
        """
        # Sum the vectors of first shell neighbors
        first_shell = (nn_shells == 1) * 1
        normal = np.sum(nn_vectors * first_shell[:, :, np.newaxis], axis=1)
        # Handle zero normals
        norms = np.linalg.norm(normal, axis=1)
        zero_norms = norms == 0
        norms[zero_norms] = np.finfo(float).eps
        # Normalize
        normal = normal / norms[:, np.newaxis]
        return normal

    @staticmethod
    def unitnormals_relaxed(nn_vectors, first_shell_idxs):
        """
        Compute unit normal vectors for each reference point,
        considering only the first shell of neighbors.

        Parameters
        ----------
        nn_vectors : list of numpy.ndarray
            The nearest neighbor vectors for each reference point.
        first_shell_idxs : numpy.ndarray
            The index of the last vector belonging to the first shell for each reference point.

        Returns
        -------
        numpy.ndarray
            The unit normal vectors for each reference point.
        """
        normal = np.zeros((len(nn_vectors), nn_vectors[0].shape[1]))
        for i, nnv in enumerate(nn_vectors):
            normal[i] = np.sum(nnv[: first_shell_idxs[i], :], axis=0)
        # Normalize
        norms = np.linalg.norm(normal, axis=1)
        zero_norms = norms == 0
        norms[zero_norms] = np.finfo(float).eps
        normal = normal / norms[:, np.newaxis]
        return normal

    def charge_distribution_z(self, config=None):
        """
        Simulates charge distribution on the surface of a structure
        under an external electric field.

        Parameters
        ----------
        config : SimulationConfig, optional
            Configuration parameters for the simulation.

        Returns
        -------
        dict
            A dictionary containing initial and final charge distributions,
            surface indices, and other relevant information.
        """
        if config is None:
            config = SimulationConfig()

        structure = self.structure

        # Identify surface atoms
        if config.relaxed:
            nn = structure.get_neighbors_by_distance()
            num_first_shell_neighbors = [
                np.sum(array < config.nd) for array in nn.distances
            ]
            surf_indices = np.where(np.array(num_first_shell_neighbors) < config.nn_n)[
                0
            ]
            un = self.unitnormals_relaxed(
                nn_vectors=nn.vecs,
                first_shell_idxs=np.array(num_first_shell_neighbors) - 1,
            )
        else:
            nn = structure.get_neighbors(num_neighbors=24)
            coord_numbers = np.apply_along_axis(self.nunique, 1, nn.shells)
            surf_indices = np.where(np.array(coord_numbers) < config.nn_n)[0]
            un = self.unitnormals(nn_vectors=nn.vecs, nn_shells=nn.shells)

        # Filter atoms below zheight
        zheight_ind = np.where(structure.positions[surf_indices, 2] < config.zheight)
        surf_un = un[surf_indices]

        # Constants and total charge
        e_field_converted = self.e_field / 51.4  # Convert e_field
        a_to_bohr = 1.8897  # Angstrom to Bohr conversion factor
        area = 2 * np.pi * config.radius**2 * a_to_bohr
        total_charge = e_field_converted * area / (4 * np.pi)

        # Initialize charges
        initial_charge = np.random.random(len(surf_indices))
        initial_charge[zheight_ind] = 0.0
        initial_charge = (initial_charge / np.sum(initial_charge)) * total_charge
        current_charge = initial_charge.copy()

        # Prepare position arrays
        surf_positions = structure.positions[surf_indices]
        num_surf_atoms = len(surf_indices)

        # Begin iterative charge distribution calculation
        for step in tqdm(range(config.steps), desc="Charge Distribution Convergence"):
            # Compute position differences between all pairs
            # pos_vectors[i, j, :] = surf_positions[j, :] - surf_positions[i, :]
            pos_vectors = (
                surf_positions[np.newaxis, :, :] - surf_positions[:, np.newaxis, :]
            )

            # Compute norms
            norms = np.linalg.norm(pos_vectors, axis=2)
            zero_norms = norms == 0
            norms[zero_norms] = np.finfo(float).eps

            # Compute pos_vectors_norm
            pos_vectors_norm = 1 / norms**3
            pos_vectors_norm[zero_norms] = 0.0

            # Compute dot products between position vectors and unit normals at atom i
            # For each i, dotprod[i, :] = np.dot(pos_vectors[i, :, :], surf_un[i, :])
            surf_un_expanded = surf_un[
                :, np.newaxis, :
            ]  # Shape: (num_surf_atoms, 1, 3)
            dotprod = np.sum(
                pos_vectors * surf_un_expanded, axis=2
            )  # Shape: (num_surf_atoms, num_surf_atoms)

            # Compute charge updates
            # For each i, new_charge[i] = (1 / (2π)) * Σ_j [current_charge[j] * dotprod[i, j] * pos_vectors_norm[i, j]]
            charge_matrix = current_charge[np.newaxis, :] * dotprod * pos_vectors_norm
            new_charge = (1 / (2 * np.pi)) * np.sum(charge_matrix, axis=1)

            # Zero out charges for indices in zheight_ind
            new_charge[zheight_ind[0]] = 0.0

            # Calculate saturation factor
            sat = np.sum(initial_charge) / np.sum(new_charge)
            new_charge *= sat

            # Check for convergence
            if np.linalg.norm(new_charge - current_charge) < config.epsilon:
                logging.info("Convergence reached at step %d", step)
                break

            current_charge = new_charge.copy()
        else:
            logging.warning("Convergence not reached after %d steps", config.steps)

        # Calculate Maxwell stress
        maxwell_stress = new_charge**2 / (2 * sat**2)

        # Store results in instance variables
        self.charge_distribution = new_charge
        self.surface_indices = surf_indices
        self.surface_normals = surf_un

        output = {
            "initial_charge": initial_charge,
            "final_charge": new_charge,
            "S_at": sat,
            "maxwell_stress": maxwell_stress,
            "surface_indices": surf_indices,
            "surface_neutral_indices": zheight_ind,
        }
        return output

    def evaporation_trajectory(self, evap_ind, num_steps=100, dt=1):
        """
        Simulates the trajectory of an atom evaporating from the surface.

        Parameters
        ----------
        evap_ind : int
            The index of the evaporating atom in the surface indices array.
        num_steps : int, optional
            The number of simulation steps for the evaporation trajectory.
        dt : float, optional
            The time step size for the simulation.

        Returns
        -------
        numpy.ndarray
            The evaporation trajectory of the atom.
        """
        positions = self.structure.positions
        surf_indices = self.surface_indices
        new_charge = self.charge_distribution

        evap_trajectory = [positions[surf_indices[evap_ind]]]

        for i in range(num_steps):
            if i == 0:
                pos_t = positions[surf_indices[evap_ind]]
                pos_vectors = pos_t - positions[surf_indices]
                norms = np.linalg.norm(pos_vectors, axis=1)
                zero_norms = norms == 0
                norms[zero_norms] = np.finfo(float).eps
                pos_vectors_norm = 1 / norms**3
                pos_vectors_norm[evap_ind] = 0.0
                coulomb_contributions = (
                    new_charge[evap_ind] * new_charge * pos_vectors_norm
                )
                force = (1 / (4 * np.pi)) * np.sum(
                    pos_vectors * coulomb_contributions[:, np.newaxis], axis=0
                )
                next_pos = evap_trajectory[0] + force * dt * dt
                evap_trajectory.append(next_pos)
            else:
                pos_t = evap_trajectory[i]
                pos_vectors = pos_t - positions[surf_indices]
                norms = np.linalg.norm(pos_vectors, axis=1)
                zero_norms = norms == 0
                norms[zero_norms] = np.finfo(float).eps
                pos_vectors_norm = 1 / norms**3
                coulomb_contributions = (
                    new_charge[evap_ind] * new_charge * pos_vectors_norm
                )
                force = (1 / (4 * np.pi)) * np.sum(
                    pos_vectors * coulomb_contributions[:, np.newaxis], axis=0
                )
                next_pos = (
                    (2 * evap_trajectory[i])
                    - evap_trajectory[i - 1]
                    + (0.5 * force * dt**2)
                )
                evap_trajectory.append(next_pos)
        return np.array(evap_trajectory)

    @staticmethod
    def detector_image(evaporation_trajectory, det_z=1e9):
        """
        Projects the evaporation trajectory onto a detector plane.

        Parameters
        ----------
        evaporation_trajectory : numpy.ndarray
            The trajectory of the evaporating atom.
        det_z : float, optional
            The z-coordinate of the detector plane.

        Returns
        -------
        numpy.ndarray
            The projected position of the atom on the detector plane.
        """
        vector = evaporation_trajectory[-1] - evaporation_trajectory[-2]
        if vector[2] == 0:
            logging.warning("Trajectory is parallel to the detector plane.")
            return None
        ratio = (det_z - evaporation_trajectory[-1][2]) / vector[2]
        projection = evaporation_trajectory[-1] + ratio * vector
        projection[2] = det_z
        return projection

    def run_evaporation(self, num_atoms=10, path=None, config=None, **kwargs):
        """
        Runs the evaporation process simulation for a specified number of atoms.

        Parameters
        ----------
        num_atoms : int, optional
            The number of atoms to simulate evaporation for.
        path : str, optional
            Path to save the output data.
        config : SimulationConfig, optional
            Configuration parameters for the simulation.
        **kwargs : dict, optional
            Additional keyword arguments to configure the simulation.

        Returns
        -------
        None
        """
        if path is None:
            path = os.getcwd()
        if config is None:
            config = SimulationConfig()

        time_step = kwargs.get("dt", 1.5)
        traj_steps = kwargs.get("num_steps", 200)

        # Initial charge distribution
        self.charge_distribution_z(config=config)

        fin_evapos = {}
        tip_pos = {}
        tip_pos_charge = {}
        tip_surf_ind_pos = {}
        impact_positions_list = []

        structure = self.structure.copy()

        for i in tqdm(range(num_atoms), desc="Evaporation Simulations"):
            # Identify atom with highest Maxwell stress
            maxwell_stress = self.charge_distribution**2 / (
                2 * self.charge_distribution.sum() ** 2
            )
            evap_ind = np.argmax(maxwell_stress)
            # Simulate evaporation trajectory
            fin_evapo = self.evaporation_trajectory(
                evap_ind=evap_ind, num_steps=traj_steps, dt=time_step
            )
            # Remove evaporated atom
            del structure[self.surface_indices[evap_ind]]
            # Update the structure in the model
            self.structure = structure
            # Recalculate charge distribution
            self.charge_distribution_z(config=config)
            # Get impact position
            impact_position = self.detector_image(fin_evapo, det_z=1e6)
            # Store results
            fin_evapos[str(i)] = fin_evapo
            tip_pos[str(i)] = structure.positions.copy()
            tip_pos_charge[str(i)] = self.charge_distribution.copy()
            tip_surf_ind_pos[str(i)] = self.surface_indices.copy()
            impact_positions_list.append(impact_position)

        impact_positions = np.array(impact_positions_list)

        # Save data with compression
        h5io.write_hdf5(
            os.path.join(path, "fin_evapos.h5"),
            fin_evapos,
            overwrite=True,
            compression=0,
        )
        h5io.write_hdf5(
            os.path.join(path, "tip_pos.h5"), tip_pos, overwrite=True, compression=0
        )
        h5io.write_hdf5(
            os.path.join(path, "tip_pos_charge.h5"),
            tip_pos_charge,
            overwrite=True,
            compression=0,
        )
        h5io.write_hdf5(
            os.path.join(path, "tip_surf_ind.h5"),
            tip_surf_ind_pos,
            overwrite=True,
            compression=0,
        )
        h5io.write_hdf5(
            os.path.join(path, "det_coordinates.h5"),
            impact_positions,
            overwrite=True,
            compression=0,
        )

    def collect_output(self, path=None):
        """
        Collects output data from the RRModel simulation.

        Parameters
        ----------
        path : str, optional
            The path to the working directory containing the output files.

        Returns
        -------
        tuple
            A tuple containing the evaporation trajectories, tip positions, tip charges, and surface indices.
        """
        if path is None:
            path = os.getcwd()

        try:
            fin_evapos = h5io.read_hdf5(os.path.join(path, "fin_evapos.h5"))
            tip_pos = h5io.read_hdf5(os.path.join(path, "tip_pos.h5"))
            tip_pos_charge = h5io.read_hdf5(os.path.join(path, "tip_pos_charge.h5"))
            tip_surf_ind = h5io.read_hdf5(os.path.join(path, "tip_surf_ind.h5"))
            return fin_evapos, tip_pos, tip_pos_charge, tip_surf_ind
        except Exception as e:
            logging.error("Failed to read output files: %s", e)
            raise
