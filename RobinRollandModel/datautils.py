import numpy as np
from scipy.spatial import ConvexHull

def isInHull(P, hull):
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

    Note:
    The function uses the convex hull's inequalities to check if each point in P satisfies all of them,
    implying the point is inside the hull.
    """
    A = hull.equations[:, 0:-1]
    b = np.transpose(np.array([hull.equations[:, -1]]))
    is_in_hull = np.all((A @ np.transpose(P)) <= np.tile(-b, (1, len(P))), axis=0)
    return is_in_hull

def create_tip(structure, h=80, a=20, ah=20, zheight=50):
    """
    Creates and modifies the tip of a given pyiron structure object based on specified dimensions,
    simulating a probe or AFM tip.

    Parameters:
    - structure : pyiron.atomistics.structure.atoms.Atoms
        The pyiron structure object to be modified.
    - h : float
        The height of the cylindrical base of the tip.
    - a : float
        The radius of the tip base.
    - ah : float
        The radius of the hemispherical end of the tip.
    - zheight : float
        The height at which the tip starts. Currently not used in calculations.

    Returns:
    - np.ndarray
        The positions of atoms within the structure that lie within the convex hull of the tip,
        after being adjusted and flipped.

    Note:
    The function generates a tip shape by first repeating the structure to fit the tip dimensions,
    then calculating the positions for a cylindrical base and a hemispherical end. These positions
    are used to create a convex hull, and atoms within this hull are identified as part of the tip.
    The z-axis is flipped for these atoms, and their position is adjusted accordingly.
    """
    structure = structure.repeat([int(ah * 0.75), int(ah * 0.75), int(h * 0.5)])
    positions = structure.positions
    if ah == a:
        hr = ah
        u = np.linspace(0, 2 * np.pi, 70)
        v = np.linspace(hr, h, 70)
        uu, vv = np.meshgrid(u, v)
        x = (a) * np.cos(uu)
        x -= np.min(x)
        y = (a) * np.sin(uu)
        y -= np.min(y)
        z = vv
    else:
        hr = h * ah / (a)
        u = np.linspace(0, 2 * np.pi, 70)
        v = np.linspace(hr, h, 70)
        uu, vv = np.meshgrid(u, v)
        x = (a) * vv * np.cos(uu) / h
        y = (a) * vv * np.sin(uu) / h
        z = vv
    phi = np.linspace(0, 1 * np.pi, 70)
    uu, pp = np.meshgrid(u, phi)
    xh = ah * np.sin(pp) * np.cos(uu)
    xh -= np.min(xh)
    yh = ah * np.sin(pp) * np.sin(uu)
    yh -= np.min(yh)
    zh = ah * np.cos(pp)
    xh = xh[zh < 0]
    yh = yh[zh < 0]
    zh = zh[zh < 0]
    zh += hr
    tip_pos_cod = np.vstack([np.hstack([x.ravel(), xh.ravel()]), np.hstack([y.ravel(), yh.ravel()]),
                             np.hstack([z.ravel(), zh.ravel()])]).T
    hull = ConvexHull(tip_pos_cod)
    tip_ind = isInHull(positions, hull)
    lies_in_tip = positions[tip_ind]
    lies_in_tip = np.matmul(lies_in_tip, [[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    lies_in_tip[:, 2] -= np.min(lies_in_tip[:, 2])
    return lies_in_tip
