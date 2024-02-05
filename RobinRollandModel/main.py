import numpy as np


def nunique(arr):
    _,r = np.unique(arr,return_counts=True)
    return r[0]

def unitnormals(nn_vectors,nn_shells):
    normal=np.sum(nn_vectors*((nn_shells==1)*1)[:,:,np.newaxis],axis=1)
    np.seterr(divide='ignore')
    normal= np.divide(normal,np.linalg.norm(normal,axis=1)[:,np.newaxis])
    return normal  

def unitnormals_relaxed(nn_vectors, first_shell_idxs):
    """
    Compute unit normal vectors for each reference point.

    Parameters
    ----------
    nn_vectors : list of numpy.ndarray, each with shape (k_i, D)
        The nearest neighbor vectors for each reference point.
    first_shell_idxs : numpy.ndarray, shape (N,)
        The index of the last vector belonging to the first shell for each reference point.

    Returns
    -------
    numpy.ndarray, shape (N, D)
        The unit normal vectors for each reference point.
    """
    normal = np.zeros((len(nn_vectors), nn_vectors[0].shape[1]))
    for i, nnv in enumerate(nn_vectors):
        normal[i] = np.sum(nnv[:first_shell_idxs[i], :], axis=0)
    np.seterr(divide='ignore')
    normal = np.divide(normal, np.linalg.norm(normal, axis=1)[:, np.newaxis])
    return normal

def charge_distribution_z(structure,e_field=4,radius=20,steps=1000,epsilon=1e-9, zheight=50,nn_n=12, relaxed=False,nd=3.):
    if relaxed:
        nn = structure.get_neighbors_by_distance()
        nn_ = [np.sum(array < nd) for array in nn.distances]
        surf_indices = np.where(np.array(nn_)<nn_n)[0]
        un = unitnormals_relaxed(nn_vectors=nn.vecs,first_shell_idxs=np.array(nn_)-1)
        zheight_ind = np.where(structure.positions[surf_indices,2]<zheight)  
        surf_un = un[surf_indices]   
    else:
        nn = structure.get_neighbors(num_neighbors=24)
        coord = np.apply_along_axis(nunique,1,nn.shells)
        surf_indices = np.where(np.array(coord)<nn_n)[0]
        zheight_ind = np.where(structure.positions[surf_indices,2]<zheight)
        un = unitnormals(nn_vectors=nn.vecs,nn_shells=nn.shells)
        surf_un = un[surf_indices]

    ANGSTROM_TO_BOHR = 1.8897
    right_field = e_field/ 51.4
    area = 2*np.pi*radius**2* ANGSTROM_TO_BOHR
    total_charge = (right_field ) * area / (4 * np.pi)
    np.seterr(divide='ignore')
    initial_charge = np.random.random(len(structure[surf_indices]))
    initial_charge[zheight_ind]=0
    initial_charge = (initial_charge/np.sum(initial_charge))*total_charge
    initial_charge_2 = initial_charge
    for n in range(steps):
        new_charge = []
        for i in range(len(surf_indices)):
            if i in list(zheight_ind)[0]:
                new_charge.append(0.0)
            else:
                np.seterr(divide='ignore')
                pos_vectors = structure[surf_indices].positions - structure.positions[surf_indices[i]]
                dotprod = np.dot(pos_vectors,surf_un[i])
                pos_vectors_norm = 1/np.linalg.norm(pos_vectors,axis=1)**3
                pos_vectors_norm[i]=0
                charge_sat = 1/(2*np.pi) * np.sum(initial_charge_2*dotprod*pos_vectors_norm)
                new_charge.append(charge_sat)
        sat = np.sum(initial_charge)/np.sum(new_charge)
        new_charge = np.array(new_charge)
        new_charge = new_charge*sat 
        initial_charge_2 = np.array(new_charge)
        if np.linalg.norm(new_charge-initial_charge_2) < epsilon:
            break
        else:
            print("convergence not reached")
    maxwell_stress = new_charge**2/(2*sat**2)
    output = dict()
    output['initial_charge']=initial_charge
    output['final_charge'] = new_charge
    output['S_at']=sat
    output['maxwell_stress']=maxwell_stress
    output['surface_indices']=surf_indices
    output['surface_neutral_indices']=zheight_ind
    return output

def evaporation_trajectory(structure,surf_indices,new_charge,evap_ind=585,num_steps=100,dt=1):
    evap_trajectory = [structure.positions[surf_indices[evap_ind]]]
    for i in range(num_steps):
        if i == 0:
            pos_t = structure.positions[surf_indices[evap_ind]]
            pos_vectors = pos_t - structure[surf_indices].positions
            pos_vectors_norm = 1/np.linalg.norm(pos_vectors,axis=1)**3
            pos_vectors_norm[evap_ind]=0
            aa = new_charge[evap_ind]*new_charge*pos_vectors_norm
            force = 1/(4*np.pi) * np.sum(pos_vectors*aa[:,np.newaxis],axis=0)
            next_pos = evap_trajectory[0] + force*dt*dt
            evap_trajectory.append(next_pos)
        else:
            pos_t = evap_trajectory[i-1]
            pos_vectors = pos_t - structure[surf_indices].positions
            pos_vectors_norm = 1/np.linalg.norm(pos_vectors,axis=1)**3
            pos_vectors_norm[evap_ind]=0
            aa = new_charge[evap_ind]*new_charge*pos_vectors_norm
            force = 1/(4*np.pi) * np.sum(pos_vectors*aa[:,np.newaxis],axis=0)
            next_pos = 2*evap_trajectory[i] - evap_trajectory[i-1] + (0.5*force*dt**2)
            evap_trajectory.append(next_pos)
    return np.array(evap_trajectory)


def detecttor_image(evaporation_trajectory,det_z = 1e9):
    vector = evaporation_trajectory[-2]-evaporation_trajectory[-1]
    normal = [0,0,-1]
    det_center = [25,25,det_z]
    projection = project_vector_onto_plane(vector,normal,det_center)
    return projection

def project_vector_onto_plane(vector, normal, point_on_plane):
    # Calculate the projection
    vector = np.array(vector)
    normal = np.array(normal)
    point_on_plane = np.array(point_on_plane)
    
    projection = vector - np.dot(vector - point_on_plane, normal) * normal
    
    return projection


def coulomb_force(positions,new_charge=None, surf_indices=None):
    """
    Takes the equilibrium charge (new_charge) on surface(surf_indices), for the structure positions.
    
    returns the coulomb forces (c_f) of shape positions
    """
    force = []
    for k in range(len(new_charge)):
        pos_vectors = positions[surf_indices] - positions[surf_indices[k]]
        pos_vectors_norm = 1/np.linalg.norm(pos_vectors,axis=1)**3
        coulomb_force = pos_vectors*(new_charge * new_charge[k] * pos_vectors_norm)[:,np.newaxis] 
        coulomb_force = np.delete(coulomb_force,k, axis=0)
        force.append(np.sum(coulomb_force,axis=0))
    c_f = np.zeros_like(positions)
    c_f[surf_indices] = np.stack(force)
    return c_f

def run_evaporation(structure,e_field=4,num_atoms=500):
    tip_output = charge_distribution_z(structure=structure,e_field=e_field,radius=20,steps=1000,epsilon=1e-9)
    fin_evapos = {}
    tip_pos={}
    tip_pos_charge={}
    tip_surf_ind_pos= {}
    # for i in range(len(tip_output['final_charge'].nonzero()[0])):
    for i in range(num_atoms):
        evap_ind = np.argmax(tip_output['maxwell_stress'])
        fin_evapos[i] = evaporation_trajectory(structure=structure,surf_indices=tip_output['surface_indices'],
                                            new_charge = tip_output['final_charge'],
                                            evap_ind=evap_ind,num_steps=200,dt=1.5)
        del structure[tip_output['surface_indices'][evap_ind]]
        new_structure = structure.copy()
        tip_output = charge_distribution_z(structure=new_structure,e_field=e_field,radius=20,
                                    steps=1000,epsilon=1e-9,zheight=50)
        tip_pos[i] = new_structure
        tip_pos_charge[i] = tip_output['final_charge']
        tip_surf_ind_pos[i] = tip_output['surface_indices']

    return tip_pos,tip_pos_charge,tip_surf_ind_pos

