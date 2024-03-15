"""Module for creating tips for RRmodel module"""

import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import h5io


class TipGenerator:
    """Class for creating tips from a basic structure"""

    def __init__(self, structure, h=80, ah=20, alpha=None, zheight=50):
        """
        Initializes the TipGenerator with the structure and dimensions for the tip.

        Parameters:
        - structure : pyiron.atomistics.structure.atoms.Atoms
            The pyiron structure object to be modified.
        - h : float
            The height of the cylindrical base of the tip. if alpha is not none, then
            h is the total height of the tip
        - alpha : float, optional
            Shank angle in degrees.
        - ah : float
            The radius of the hemispherical end of the tip.
        - zheight : float
            The height at which the tip starts. Currently not used in calculations.
        """
        self.structure = structure
        self.h = h
        self.a = ah
        self.ah = ah
        self.zheight = zheight
        self.alpha = alpha

    def is_in_hull(self, points, hull):
        """
        Determine if the list of points P lies inside the given convex hull.

        Parameters:
        - P : np.ndarray
            Array of points to check, where each row represents a point in n-dimensional space.
        - hull : scipy.spatial.ConvexHull
            A ConvexHull object representing the convex hull within 
            which the points are to be checked.

        Returns:
        - np.ndarray
            A boolean array where True indicates the point is inside the convex hull.
        """
        a = hull.equations[:, 0:-1]
        b = np.transpose(np.array([hull.equations[:, -1]]))
        is_in_hull = np.all((a @ points.T) + b <= 0, axis=0)
        return is_in_hull

    @staticmethod
    def create_shank_tip(
        radius_spherical_cap, shank_angle_degrees, total_height, num_points
    ):
        """Create a tip with shank."""
        # Convert shank angle from degrees to radians
        shank_angle = np.radians(shank_angle_degrees)
        radius_top_base = radius_spherical_cap * np.cos(shank_angle)

        # Determine the height of the spherical cap
        h_spherical_cap = radius_spherical_cap - radius_spherical_cap * np.sin(
            shank_angle
        )

        # Determine the height of the truncated cone
        h_truncated_cone = total_height - h_spherical_cap

        # Spherical cap coordinates
        theta_cap = np.linspace(0, 2 * np.pi, num_points)
        phi_cap = np.linspace(0, np.pi / 2 - shank_angle, num_points)
        Theta_cap, Phi_cap = np.meshgrid(theta_cap, phi_cap)
        x_cap = radius_spherical_cap * np.sin(Phi_cap) * np.cos(Theta_cap)
        y_cap = radius_spherical_cap * np.sin(Phi_cap) * np.sin(Theta_cap)
        z_cap = radius_spherical_cap * np.cos(Phi_cap)
        x_cap = x_cap.ravel()
        y_cap = y_cap.ravel()
        z_cap = z_cap.ravel()

        z_cap -= np.min(z_cap)
        z_cap += h_truncated_cone
        # Calculate the radius at the base (R1) based on Radius of spherical end and the shank angle
        R1 = radius_top_base + h_truncated_cone * np.tan(shank_angle)

        # Create arrays for u and v values
        u_values = np.linspace(0, 1, num_points)
        v_values = np.linspace(0, 2 * np.pi, num_points)

        # Create meshgrid for u and v
        u, v = np.meshgrid(u_values, v_values)

        # Calculate x, y, and z coordinates using parametric equations
        x = (R1 + (radius_top_base - R1) * u) * np.cos(v)
        y = (R1 + (radius_top_base - R1) * u) * np.sin(v)
        z = h_truncated_cone * u

        x_cone = x.ravel()
        y_cone = y.ravel()
        z_cone = z.ravel()

        # Adjust z_cone to start from bottom
        z_cone_adjusted = z_cone

        # Combine point clouds for cone and cap
        x_cloud = np.concatenate((x_cone, x_cap))
        y_cloud = np.concatenate((y_cone, y_cap))
        z_cloud = np.concatenate((z_cone_adjusted, z_cap))
        x_cloud -= np.min(x_cloud)  #
        y_cloud -= np.min(y_cloud)  #

        return x_cloud, y_cloud, z_cloud, R1

    def create_tip(self):
        """
        Creates and modifies the tip of the given pyiron structure object based on specified dimensions,
        simulating a APT tip.

        Returns:
        - np.ndarray
            The positions of atoms within the structure that lie within the convex hull of the tip,
            after being adjusted and flipped.
        """

        if self.alpha is None:
            # structure = self.structure.repeat([int(self.ah * 0.75), int(self.ah * 0.75), int(self.h * 0.5)])
            lp = self.structure.get_cell()[0][0] / 2
            lpz = self.structure.get_cell()[2][2]
            structure = self.structure.repeat(
                [
                    int(np.ceil((self.ah + 1) / lp)),
                    int(np.ceil((self.ah + 1) / lp)),
                    int(np.ceil((self.h + 1) / lpz)),
                ]
            )
            positions = structure.positions
            hr = self.ah
            u = np.linspace(0, 2 * np.pi, 70)
            v = np.linspace(hr, self.h, 70)
            uu, vv = np.meshgrid(u, v)
            x = (self.a) * np.cos(uu)
            x -= np.min(x)
            y = (self.a) * np.sin(uu)
            y -= np.min(y)
            z = vv
            phi = np.linspace(0, 1 * np.pi, 70)
            uu, pp = np.meshgrid(u, phi)
            xh = self.ah * np.sin(pp) * np.cos(uu)
            xh -= np.min(xh)
            yh = self.ah * np.sin(pp) * np.sin(uu)
            yh -= np.min(yh)
            zh = self.ah * np.cos(pp)
            xh = xh[zh < 0]
            yh = yh[zh < 0]
            zh = zh[zh < 0]
            zh += hr
            tip_pos_cod = np.vstack(
                [
                    np.hstack([x.ravel(), xh.ravel()]),
                    np.hstack([y.ravel(), yh.ravel()]),
                    np.hstack([z.ravel(), zh.ravel()]),
                ]
            ).T
            hull = ConvexHull(tip_pos_cod)
            tip_ind = self.is_in_hull(positions, hull)
            lies_in_tip = positions[tip_ind]
            lies_in_tip = np.matmul(lies_in_tip, [[1, 0, 0], [0, 1, 0], [0, 0, -1]])
            lies_in_tip[:, 2] -= np.min(lies_in_tip[:, 2])
        else:
            x, y, z, _ = TipGenerator.create_shank_tip(
                radius_spherical_cap=self.ah,
                shank_angle_degrees=self.alpha,
                total_height=self.h,
                num_points=70,
            )
            # structure = self.structure.repeat([int(R1 * 0.75), int(R1 * 0.75), int(self.h * 0.5)])
            lp = self.structure.get_cell()[0][0]
            lpz = self.structure.get_cell()[2][2]
            structure = self.structure.repeat(
                [
                    int(np.ceil((np.max(x) + 5) / lp)),
                    int(np.ceil((np.max(y) + 5) / lp)),
                    int(np.ceil((np.max(z) + 1) / lpz)),
                ]
            )
            positions = structure.positions
            tip_pos_cod = np.vstack([x, y, z]).T
            hull = ConvexHull(tip_pos_cod)
            tip_ind = self.is_in_hull(positions, hull)
            lies_in_tip = positions[tip_ind]
            lies_in_tip = np.matmul(lies_in_tip, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            lies_in_tip[:, 2] -= np.min(lies_in_tip[:, 2])
        elements = structure.get_species_symbols()[structure.get_chemical_indices()][
            tip_ind
        ]
        return lies_in_tip, elements

    def create_tip_pyiron(self, pr):
        """Creates  a pyiron structure for the tip"""
        lies_in_tip, elements = self.create_tip()
        struc_tip = pr.create.structure.atoms(elements=elements, positions=lies_in_tip)
        # cell = [[np.max(struc_tip.positions[:,0])+15,np.min(struc_tip.positions[:,1])-15,0],[np.min(struc_tip.positions[:,0])-15,np.max(struc_tip.positions[:,1])+15,0],[0,0,np.max(struc_tip.positions[:,2])+40]]
        cell = [
            np.max(struc_tip.positions[:, 0]) + 5,
            np.max(struc_tip.positions[:, 1]) + 5,
            np.max(struc_tip.positions[:, 2]) + 40,
        ]
        struc_tip.cell = cell
        return struc_tip


def visualize(structure=None, charge=None, surf_indices=None):
    """Visualize using plotly"""
    if charge is None:
        try:
            charge = structure.indices
        except AttributeError:
            charge = np.zeros(len(structure))
    if surf_indices is None:
        surf_indices = np.arange(len(structure))

    try:
        positions = structure.get_positions()
    except AttributeError:
        positions = structure

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=positions[surf_indices, 0],
                y=positions[surf_indices, 1],
                z=positions[surf_indices, 2],
                mode="markers",
                marker=dict(
                    size=9,
                    color=charge,  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    line=dict(color="MediumPurple", width=12),
                ),
            )
        ]
    )
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def visualize_evaporation(
    tip_pos, tip_pos_charge, tip_surf_ind_pos, fin_evapos, ind, ind_all=False
):
    """
    Visualizes the positions of atoms before and after evaporation using Plotly.

    Parameters
    ----------
    tip_pos : dict
        A dictionary containing the updated positions of the structure after each evaporation event.
    tip_pos_charge : dict
        A dictionary containing the final charge distributions post-evaporation.
    tip_surf_ind_pos : dict
        A dictionary containing the surface indices post-evaporation.
    ind : int
        Index of the evaporating atom
    """
    fig = go.Figure()
    if ind_all:
        fin_ind = len(tip_pos) - 1
        initial_positions = tip_pos[
            fin_ind
        ]  # Adjust this according to how you store positions in `tip_pos`.
        charge = tip_pos_charge[fin_ind]
        surf_ind = tip_surf_ind_pos[fin_ind]
        fig.add_trace(
            go.Scatter3d(
                x=initial_positions[surf_ind, 0],
                y=initial_positions[surf_ind, 1],
                z=initial_positions[surf_ind, 2],
                mode="markers",
                name="Initial Positions",
                marker=dict(size=2, color=charge),
            )
        )

        # Loop through each evaporation event to visualize the end positions
        for i in range(len(tip_pos)):
            # Adjust this according to your data structure
            fig.add_trace(
                go.Scatter3d(
                    x=fin_evapos[i][:, 0],
                    y=fin_evapos[i][:, 1],
                    z=fin_evapos[i][:, 2],
                    mode="markers",
                    name=f"Final Positions {i}",
                    marker=dict(size=2, color="red"),
                )
            )
        fig.update_layout(
            title="Evaporation Simulation Visualization",
            scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig.show()
        return

    # Assuming the structure positions are stored in 
    # a numpy array format within the 'tip_pos' dictionary.
    # Here, we visualize the initial structure before evaporation.
    initial_positions = tip_pos[
        ind
    ]  # Adjust this according to how you store positions in `tip_pos`.
    charge = tip_pos_charge[ind]
    surf_ind = tip_surf_ind_pos[ind]
    fig.add_trace(
        go.Scatter3d(
            x=initial_positions[surf_ind, 0],
            y=initial_positions[surf_ind, 1],
            z=initial_positions[surf_ind, 2],
            mode="markers",
            name="Initial Positions",
            marker=dict(size=2, color=charge),
        )
    )

    # Loop through each evaporation event to visualize the end positions
    # Adjust this according to your data structure
    fig.add_trace(
        go.Scatter3d(
            x=fin_evapos[ind][:, 0],
            y=fin_evapos[ind][:, 1],
            z=fin_evapos[ind][:, 2],
            mode="markers",
            name="Final Positions",
            marker=dict(size=2, color="red"),
        )
    )

    # Customize layout
    fig.update_layout(
        title="Evaporation Simulation Visualization",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()


def collect_output(path=None):
    """collects the output to numpy arrays from h5 files"""
    fin_evapos = h5io.read_hdf5(f'{path}/fin_evapos.h5')
    tip_pos = h5io.read_hdf5(f'{path}/tip_pos.h5')
    tip_pos_charge = h5io.read_hdf5(f'{path}/tip_pos_charge.h5')
    tip_surf_ind = h5io.read_hdf5(f'{path}/tip_surf_ind.h5')
    return fin_evapos, tip_pos, tip_pos_charge, tip_surf_ind
