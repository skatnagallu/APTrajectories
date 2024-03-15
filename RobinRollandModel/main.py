"""Module for equilibrium charge distribution based on the RR model"""
import os
import numpy as np
import h5io
from tqdm import trange


class RRModel:
    """
    A class representing the RRModel for simulating charge distribution, 
    evaporation trajectory, and other related physical processes on a 
    nanostructure's surface under an external electric field.
    """

    def __init__(self, tip_generator, structure, e_field):
        """
        Initializes the RRModel instance.
        Parameters
        ----------
        tip_generator: object
            An instance of TipGenerator class in datautils
        structure : object
            An object representing the nanostructure.
        e_field : float, optional
            The external electric field magnitude.
        """
        self.tip_generator = (
            tip_generator  # instance of TipGenerator class used to create structure
        )
        self.structure = structure
        self.e_field = e_field

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
        normal = np.sum(nn_vectors * ((nn_shells == 1) * 1)[:, :, np.newaxis], axis=1)
        normal[np.linalg.norm(normal, axis=1) == 0] = 1e-12
        normal = np.divide(normal, np.linalg.norm(normal, axis=1)[:, np.newaxis])
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
        normal = np.divide(normal, np.linalg.norm(normal, axis=1)[:, np.newaxis])
        return normal

    @staticmethod
    def charge_distribution_z(
        structure,
        e_field=4,
        radius=20,
        steps=1000,
        epsilon=1e-9,
        zheight=50,
        nn_n=12,
        relaxed=False,
        nd=3.0,
    ):
        """
        Simulates charge distribution on the surface of a structure 
        under an external electric field.

        Parameters
        ----------
        structure : object
            An object representing the nanostructure, 
            must have `get_neighbors_by_distance` or `get_neighbors` method.
        e_field : float, optional
            The magnitude of the external electric field.
        radius : float, optional
            The radius of the structure for calculating the surface area.
        steps : int, optional
            The number of steps for the charge distribution simulation to converge.
        epsilon : float, optional
            The convergence criterion.
        zheight : float, optional
            The height limit to consider surface atoms for charge distribution.
        nn_n : int, optional
            The number of nearest neighbors to consider for determining surface atoms.
        relaxed : bool, optional
            Whether to use a relaxed criterion for surface atom determination.
        nd : float, optional
            The distance criterion for determining the first shell in the relaxed approach.

        Returns
        -------
        dict
            A dictionary containing initial and final charge distributions, 
            surface indices, and other relevant information.
        """
        if relaxed:
            nn = structure.get_neighbors_by_distance()
            nn_ = [np.sum(array < nd) for array in nn.distances]
            surf_indices = np.where(np.array(nn_) < nn_n)[0]
            un = RRModel.unitnormals_relaxed(
                nn_vectors=nn.vecs, first_shell_idxs=np.array(nn_) - 1
            )
            zheight_ind = np.where(structure.positions[surf_indices, 2] < zheight)
            surf_un = un[surf_indices]
        else:
            nn = structure.get_neighbors(num_neighbors=24)
            coord = np.apply_along_axis(RRModel.nunique, 1, nn.shells)
            surf_indices = np.where(np.array(coord) < nn_n)[0]
            zheight_ind = np.where(structure.positions[surf_indices, 2] < zheight)
            un = RRModel.unitnormals(nn_vectors=nn.vecs, nn_shells=nn.shells)
            surf_un = un[surf_indices]

        a_to_bohr = 1.8897
        right_field = e_field / 51.4
        area = 2 * np.pi * radius**2 * a_to_bohr
        total_charge = (right_field) * area / (4 * np.pi)
        initial_charge = np.random.random(len(structure[surf_indices]))
        initial_charge[zheight_ind] = 0
        initial_charge = (initial_charge / np.sum(initial_charge)) * total_charge
        initial_charge_2 = initial_charge
        step = 0
        while step < steps:
            new_charge = []
            for i,_ in enumerate(surf_indices):
                if i in list(zheight_ind)[0]:
                    new_charge.append(0.0)
                else:
                    pos_vectors = (
                        structure[surf_indices].positions
                        - structure.positions[surf_indices[i]]
                    )
                    dotprod = np.dot(pos_vectors, surf_un[i])
                    pos_vectors[np.linalg.norm(pos_vectors, axis=1) == 0] = 1e-12
                    pos_vectors_norm = 1 / np.linalg.norm(pos_vectors, axis=1) ** 3
                    pos_vectors_norm[i] = 0
                    charge_sat = (
                        1
                        / (2 * np.pi)
                        * np.sum(initial_charge_2 * dotprod * pos_vectors_norm)
                    )
                    new_charge.append(charge_sat)
            sat = np.sum(initial_charge) / np.sum(new_charge)
            new_charge = np.array(new_charge)
            new_charge = new_charge * sat
            initial_charge_2 = np.array(new_charge)
            step +=1
            if np.linalg.norm(new_charge - initial_charge_2) < epsilon:
                break
            print("convergence not reached")
        maxwell_stress = new_charge**2 / (2 * sat**2)
        output = {}
        output["initial_charge"] = initial_charge
        output["final_charge"] = new_charge
        output["S_at"] = sat
        output["maxwell_stress"] = maxwell_stress
        output["surface_indices"] = surf_indices
        output["surface_neutral_indices"] = zheight_ind
        return output

    @staticmethod
    def evaporation_trajectory(
        structure, surf_indices, new_charge, evap_ind=585, num_steps=100, dt=1
    ):
        """
        Simulates the trajectory of an atom evaporating from the surface.

        Parameters
        ----------
        structure : object
            An object representing the nanostructure, must have `positions` attribute.
        surf_indices : numpy.ndarray
            Indices of surface atoms in the structure.
        new_charge : numpy.ndarray
            The charge distribution on the surface atoms.
        evap_ind : int, optional
            The index of the evaporating atom.
        num_steps : int, optional
            The number of simulation steps for the evaporation trajectory.
        dt : float, optional
            The time step size for the simulation.

        Returns
        -------
        numpy.ndarray
            The evaporation trajectory of the atom.
        """
        evap_trajectory = [structure.positions[surf_indices[evap_ind]]]
        for i in range(num_steps):
            if i == 0:
                pos_t = structure.positions[surf_indices[evap_ind]]
                pos_vectors = pos_t - structure[surf_indices].positions
                pos_vectors[np.linalg.norm(pos_vectors, axis=1) == 0] = 1e-12
                pos_vectors_norm = 1 / np.linalg.norm(pos_vectors, axis=1) ** 3
                pos_vectors_norm[evap_ind] = 0
                aa = new_charge[evap_ind] * new_charge * pos_vectors_norm
                force = (
                    1 / (4 * np.pi) * np.sum(pos_vectors * aa[:, np.newaxis], axis=0)
                )
                next_pos = evap_trajectory[0] + force * dt * dt
                evap_trajectory.append(next_pos)
            else:
                pos_t = evap_trajectory[i - 1]
                pos_vectors = pos_t - structure[surf_indices].positions
                pos_vectors[np.linalg.norm(pos_vectors, axis=1) == 0] = 1e-12
                pos_vectors_norm = 1 / np.linalg.norm(pos_vectors, axis=1) ** 3
                pos_vectors_norm[evap_ind] = 0
                aa = new_charge[evap_ind] * new_charge * pos_vectors_norm
                force = (
                    1 / (4 * np.pi) * np.sum(pos_vectors * aa[:, np.newaxis], axis=0)
                )
                next_pos = (
                    2 * evap_trajectory[i]
                    - evap_trajectory[i - 1]
                    + (0.5 * force * dt**2)
                )
                evap_trajectory.append(next_pos)
        return np.array(evap_trajectory)

    @staticmethod
    def evaporation_trajectory_force_cut(
        structure, surf_indices, new_charge, evap_ind=585, force_cutoff=1e-4, dt=1
    ):
        """
        Simulates the trajectory of an atom evaporating from the surface.

        Parameters
        ----------
        structure : object
            An object representing the nanostructure, must have `positions` attribute.
        surf_indices : numpy.ndarray
            Indices of surface atoms in the structure.
        new_charge : numpy.ndarray
            The charge distribution on the surface atoms.
        evap_ind : int, optional
            The index of the evaporating atom.
        num_steps : int, optional
            The number of simulation steps for the evaporation trajectory.
        dt : float, optional
            The time step size for the simulation.

        Returns
        -------
        numpy.ndarray
            The evaporation trajectory of the atom.
        """
        evap_trajectory = [structure.positions[surf_indices[evap_ind]]]
        force = 1
        i = 0
        while np.linalg.norm(force) > force_cutoff:
            if i == 0:
                pos_t = structure.positions[surf_indices[evap_ind]]
                pos_vectors = pos_t - structure[surf_indices].positions
                pos_vectors[np.linalg.norm(pos_vectors, axis=1) == 0] = 1e-12
                pos_vectors_norm = 1 / np.linalg.norm(pos_vectors, axis=1) ** 3
                pos_vectors_norm[evap_ind] = 0
                aa = new_charge[evap_ind] * new_charge * pos_vectors_norm
                force = (
                    1 / (4 * np.pi) * np.sum(pos_vectors * aa[:, np.newaxis], axis=0)
                )
                next_pos = evap_trajectory[0] + force * dt * dt
                evap_trajectory.append(next_pos)
            else:
                pos_t = evap_trajectory[i - 1]
                pos_vectors = pos_t - structure[surf_indices].positions
                pos_vectors[np.linalg.norm(pos_vectors, axis=1) == 0] = 1e-12
                pos_vectors_norm = 1 / np.linalg.norm(pos_vectors, axis=1) ** 3
                pos_vectors_norm[evap_ind] = 0
                aa = new_charge[evap_ind] * new_charge * pos_vectors_norm
                force = (
                    1 / (4 * np.pi) * np.sum(pos_vectors * aa[:, np.newaxis], axis=0)
                )
                next_pos = (
                    2 * evap_trajectory[i]
                    - evap_trajectory[i - 1]
                    + (0.5 * force * dt**2)
                )
                evap_trajectory.append(next_pos)
            i += 1
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
        vector = evaporation_trajectory[-2] - evaporation_trajectory[-1]
        normal = [0, 0, -1]
        det_center = [25, 25, det_z]
        projection = RRModel.project_vector_onto_plane(vector, normal, det_center)
        return projection

    @staticmethod
    def project_vector_onto_plane(vector, normal, point_on_plane):
        """
        Projects a vector onto a plane defined by a normal vector and a point on the plane.

        Parameters
        ----------
        vector : numpy.ndarray
            The vector to project.
        normal : numpy.ndarray
            The normal vector of the plane.
        point_on_plane : numpy.ndarray
            A point on the plane.

        Returns
        -------
        numpy.ndarray
            The projection of the vector onto the plane.
        """
        vector = np.array(vector)
        normal = np.array(normal)
        point_on_plane = np.array(point_on_plane)
        projection = vector - np.dot(vector - point_on_plane, normal) * normal
        return projection

    @staticmethod
    def coulomb_force(positions, new_charge=None, surf_indices=None):
        """
            Calculates Coulomb forces between charged particles on the surface.

            Parameters
            ----------
            positions : numpy.ndarray
                The positions of all particles in the structure.
            new_charge : numpy.ndarray, optional
                The charge distribution among the particles.
            surf_indices : numpy.ndarray, optional
            Indices of particles on the surface.

        Returns
        -------
        numpy.ndarray
            The calculated Coulomb forces on each particle in the structure.
        """
        force = []
        for k,_ in enumerate(new_charge):
            pos_vectors = positions[surf_indices] - positions[surf_indices[k]]
            pos_vectors_norm = 1 / np.linalg.norm(pos_vectors, axis=1) ** 3
            coulomb_force = (
                pos_vectors
                * (new_charge * new_charge[k] * pos_vectors_norm)[:, np.newaxis]
            )
            coulomb_force = np.delete(coulomb_force, k, axis=0)
            force.append(np.sum(coulomb_force, axis=0))
        c_f = np.zeros_like(positions)
        c_f[surf_indices] = np.stack(force)
        return c_f

    @staticmethod
    def extrapolate_impact_positions(trajectories, plane_z=100):
        """
        Extrapolate the impact positions of multiple trajectories on a plane located at z=plane_z.

        Parameters:
        - trajectories: List of Numpy arrays, each of shape (n, 3), representing the trajectories.
        - plane_z: The z-coordinate of the plane.

        Returns:
        - impact_positions: Numpy array of shape (m, 3) representing the x, y, z 
        coordinates of the impact positions for m trajectories.
        """
        impact_positions = []
        for traj in trajectories:
            # Ensure trajectory has at least two points for direction vector calculation
            if traj.shape[0] >= 2:
                last_point = traj[-1]
                second_last_point = traj[-2]

                # Compute direction vector for this trajectory
                direction_vector = last_point - second_last_point

                # Compute the ratio needed to reach the plane
                if direction_vector[2] != 0:  # Avoid division by zero
                    ratio = (plane_z - last_point[2]) / direction_vector[2]
                    # Extrapolate to find the impact position
                    impact_position = last_point + ratio * direction_vector
                    impact_position[2] = plane_z  # Ensure the z-coordinate is exactly plane_z
                    impact_positions.append(impact_position)
                else:
                    # Handle trajectories parallel to the plane by appending None or a default value
                    impact_positions.append(None)
            else:
                # Handle trajectories with fewer than two points
                impact_positions.append(None)

        return np.array(impact_positions)

    def run_evaporation(self, num_atoms=10, path=None, **kwargs):
        """
        Runs the evaporation process simulation for a specified number of atoms.

        Parameters
        ----------
        num_atoms : int, optional
            The number of atoms to simulate evaporation for.
        **kwargs : dict, optional
            Additional keyword arguments to configure the simulation, 
            such as 'steps' and 'epsilon' for the charge distribution.


        Returns
        -------
        tuple
            Contains the updated structure positions, the final charge distributions,
            and surface indices post-evaporation.
        """
        if path is None:
            path = os.getcwd()
        steps = kwargs.get("steps", 1000)
        epsilon = kwargs.get("epsilon", 1e-9)
        time_step = kwargs.get("dt", 1.5)
        traj_steps = kwargs.get("num_steps", 200)

        r = self.tip_generator.ah
        zheight = self.tip_generator.zheight
        tip_output = RRModel.charge_distribution_z(
            structure=self.structure,
            e_field=self.e_field,
            radius=r,
            steps=steps,
            epsilon=epsilon,
            zheight=zheight,
        )
        fin_evapos = {}
        tip_pos = {}
        tip_pos_charge = {}
        tip_surf_ind_pos = {}
        structure = self.structure
        for i in trange(num_atoms):
            evap_ind = np.argmax(tip_output["maxwell_stress"])
            fin_evapos[str(i)] = RRModel.evaporation_trajectory(
                structure=self.structure,
                surf_indices=tip_output["surface_indices"],
                new_charge=tip_output["final_charge"],
                evap_ind=evap_ind,
                num_steps=traj_steps,
                dt=time_step,
            )
            del structure[tip_output["surface_indices"][evap_ind]]
            new_structure = structure.copy()
            tip_output = RRModel.charge_distribution_z(
                structure=new_structure,
                e_field=self.e_field,
                radius=r,
                steps=steps,
                epsilon=epsilon,
                zheight=zheight,
            )
            tip_pos[str(i)] = new_structure.positions
            tip_pos_charge[str(i)] = tip_output["final_charge"]
            tip_surf_ind_pos[str(i)] = tip_output["surface_indices"]

        impact_positions = self.extrapolate_impact_positions(list(fin_evapos.values()),1e6)

        h5io.write_hdf5(f'{path}/fin_evapos.h5',fin_evapos,overwrite=True)
        h5io.write_hdf5(f'{path}/tip_pos.h5',tip_pos,overwrite=True)
        h5io.write_hdf5(f'{path}/tip_pos_charge.h5',tip_pos_charge,overwrite=True)
        h5io.write_hdf5(f'{path}/tip_surf_ind.h5',tip_surf_ind_pos,overwrite=True)
        h5io.write_hdf5(f'{path}/det_coordinates.h5',impact_positions,overwrite=True)
