from pyiron_base.utils.error import ImportAlarm
from pyiron_base.jobs.job.template import TemplateJob
import h5py
import numpy as np
from pyiron_base.jobs.job.jobtype import JobType
try:
    from RobinRollandModel.main import RRModel
    from RobinRollandModel.datautils import TipGenerator
except ImportError:
    import_alarm = ImportAlarm("Unable to import RobinRollandmodel")

class RRModelAPTjob(TemplateJob):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input['e_field'] = 4
        self.input['tip_height'] = 80
        self.input['tip_radius'] = 20
        self.input['z_height'] = 50
        self.input['tip_shank_angle'] = None
        self.input['basic_structure'] = None
        self.input['num_atoms'] = 5 #number of atoms to evaporate

    def create_input_structure(self):
        self.input['tip_generator'] = TipGenerator(structure=self.input.basic_structure,
                                                   h=self.input.tip_height,
                                                   ah=self.input.tip_radius,
                                                   alpha=self.input.tip_shank_angle,
                                                   zheight=self.input.z_height)
        self.input['structure'] = self.input.tip_generator.create_tip_pyiron(self.project) # structure of the tip

    def run_static(self,**kwargs):
        job = RRModel(tip_generator=self.input.tip_generator,
                      structure=self.input.structure,
                      e_field=self.input.e_field)
        job.run_evaporation(num_atoms=self.input.num_atoms,**kwargs)
        #self.collect_output()

    def collect_output(self, path=None):
        if path is None:
            path = self.working_directory
        
        fin_evapos = {}
        with h5py.File(f'{path}/fin_evapos.h5','r') as output:
            for varname in output.keys ():
                atom = float(str(varname).replace('step=',''))
                fin_evapos[atom] = np.asarray(output[varname])
        self.output['evaporation_trajectories'] = fin_evapos
        
        tip_pos = {}
        with h5py.File(f'{path}/tip_pos.h5','r') as output:
            for varname in output.keys ():
                atom = float(str(varname).replace('step=',''))
                tip_pos[atom] = np.asarray(output[varname])
        self.output['tip_structures'] = tip_pos
        
        tip_pos_charge = {}
        with h5py.File(f'{path}/tip_pos_charge.h5','r') as output:
            for varname in output.keys ():
                atom = float(str(varname).replace('step=',''))
                tip_pos_charge[atom] = np.asarray(output[varname])
        self.output['equilibrium_charges'] = tip_pos_charge
        
        tip_surf_ind = {}
        with h5py.File(f'{path}/tip_surf_ind.h5','r') as output:
            for varname in output.keys ():
                atom = float(str(varname).replace('step=',''))
                tip_surf_ind[atom] = np.asarray(output[varname])
        self.output['surface_indices'] = tip_surf_ind

JobType.register(RRModelAPTjob)