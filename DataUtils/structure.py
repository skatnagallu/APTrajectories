import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from pyiron import Project
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


def isInHull(P,hull):
    '''
    Datermine if the list of points P lies inside the hull
    :return: list
    List of boolean where true means that the point is inside the convex hull
    '''
    A = hull.equations[:,0:-1]
    b = np.transpose(np.array([hull.equations[:,-1]]))
    isInHull = np.all((A @ np.transpose(P)) <= np.tile(-b,(1,len(P))),axis=0)
    return isInHull

def create_tip(structure,h=80,a=20,ah=20, zheight=50):
    """
    h is the height of the cylindrical base.
    a and ah are the radius of the tip base and the hemispherical end respecitvely.
    structure is the pyiron structure object.
    """
    structure = structure.repeat([int(ah*0.75),int(ah*0.75),int(h*0.5)])
    positions=structure.positions
    if ah == a:
        hr = ah
        u = np.linspace(0,2*np.pi,70)
        v = np.linspace(hr,h,70)
        uu,vv = np.meshgrid(u,v)
        x = (a)*np.cos(uu)
        x-=np.min(x)
        y = (a)*np.sin(uu)
        y-=np.min(y)
        z = vv
    else:
        hr = h*ah/(a)
        u = np.linspace(0,2*np.pi,70)
        v = np.linspace(hr,h,70)
        uu,vv = np.meshgrid(u,v)
        x = (a)*vv*np.cos(uu)/h
        y = (a)*vv*np.sin(uu)/h
        z = vv
    phi = np.linspace(0,1*np.pi,70)
    uu,pp = np.meshgrid(u,phi)
    xh=ah*np.sin(pp)*np.cos(uu)
    xh-=np.min(xh)
    yh=ah*np.sin(pp)*np.sin(uu)
    yh-=np.min(yh)
    zh=ah*np.cos(pp) 
    xh = xh[zh<0]
    yh = yh[zh<0]
    zh = zh[zh<0]
    zh+=hr
    tip_pos_cod = np.vstack([np.hstack([x.ravel(),xh.ravel()]),np.hstack([y.ravel(),yh.ravel()]),
                             np.hstack([z.ravel(),zh.ravel()])]).T
    hull = ConvexHull(tip_pos_cod)
    tip_ind = isInHull(positions,hull)
    lies_in_tip =positions[tip_ind]
    lies_in_tip = np.matmul(lies_in_tip,[[1,0,0],[0,1,0],[0,0,-1]])
    lies_in_tip[:,2]-=np.min(lies_in_tip[:,2])
    return lies_in_tip

