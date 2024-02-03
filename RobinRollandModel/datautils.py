import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

class TipGenerator:
    def __init__(self, structure, h=80, a=20, ah=20, zheight=50):
        """
        Initializes the TipGenerator with the structure and dimensions for the tip.

        Parameters:
        - structure : pyiron.atomistics.structure.atoms.Atoms
            The pyiron structure object to be modified.
        - h : float, optional
            The height of the cylindrical base of the tip.
        - a : float, optional
            The radius of the tip base.
        - ah : float, optional
            The radius of the hemispherical end of the tip.
        - zheight : float, optional
            The height at which the tip starts. Currently not used in calculations.
        """
        self.structure = structure
        self.h = h
        self.a = a
        self.ah = ah
        self.zheight = zheight

    def isInHull(self, P, hull):
        """
        Determine if the list of points P lies inside the given convex hull.

        Parameters:
        - P : np.ndarray
            Array of points to check, where each row represents a point in n-dimensional space.
        - hull : scipy.spatial.ConvexHull
            A ConvexHull object representing the convex hull within which the points are to be checked.

        Returns:
        - np.ndarray
            A boolean array where True indicates the point is inside the convex hull.
        """
        A = hull.equations[:, 0:-1]
        b = np.transpose(np.array([hull.equations[:, -1]]))
        is_in_hull = np.all((A @ np.transpose(P)) <= np.tile(-b, (1, len(P))), axis=0)
        return is_in_hull

    def create_tip(self):
        """
        Creates and modifies the tip of the given pyiron structure object based on specified dimensions,
        simulating a probe or AFM tip.

        Returns:
        - np.ndarray
            The positions of atoms within the structure that lie within the convex hull of the tip,
            after being adjusted and flipped.
        """
        structure = self.structure.repeat([int(self.ah * 0.75), int(self.ah * 0.75), int(self.h * 0.5)])
        positions = structure.positions
        if self.ah == self.a:
            hr = self.ah
            u = np.linspace(0, 2 * np.pi, 70)
            v = np.linspace(hr, self.h, 70)
            uu, vv = np.meshgrid(u, v)
            x = (self.a) * np.cos(uu)
            x -= np.min(x)
            y = (self.a) * np.sin(uu)
            y -= np.min(y)
            z = vv
        else:
            hr = self.h * self.ah / (self.a)
            u = np.linspace(0, 2 * np.pi, 70)
            v = np.linspace(hr, self.h, 70)
            uu, vv = np.meshgrid(u, v)
            x = (self.a) * vv * np.cos(uu) / self.h
            y = (self.a) * vv * np.sin(uu) / self.h
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
        tip_pos_cod = np.vstack([np.hstack([x.ravel(), xh.ravel()]), np.hstack([y.ravel(), yh.ravel()]),
                                 np.hstack([z.ravel(), zh.ravel()])]).T
        hull = ConvexHull(tip_pos_cod)
        tip_ind = self.isInHull(positions, hull)
        lies_in_tip = positions[tip_ind]
        lies_in_tip = np.matmul(lies_in_tip, [[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        lies_in_tip[:, 2] -= np.min(lies_in_tip[:, 2])
        return lies_in_tip

def visualize(structure=None,charge=None):
    if charge is None:
        charge = structure.indices
    fig = go.Figure(data=[go.Scatter3d(
            x=structure.positions[:,0],
            y=structure.positions[:,1],
            z=structure.positions[:,2],
            mode='markers',
            marker=dict(
                size=9,
                color=charge,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                 line=dict(
                        color='MediumPurple',
                        width=12
                )
        )
    )])
    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()