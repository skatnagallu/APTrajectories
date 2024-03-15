"""Module for pyiron job based on the RRmodel module"""
from pyiron_base.utils.error import ImportAlarm
from pyiron_base.jobs.job.template import TemplateJob
from pyiron_base.jobs.job.jobtype import JobType
import h5io
try:
    from RobinRollandModel.main import RRModel
    from RobinRollandModel.datautils import TipGenerator
except ImportError:
    import_alarm = ImportAlarm("Unable to import RobinRollandmodel")


class RRModelAPTjob(TemplateJob):
    """pyiron job class for RRModelAPT simulation"""

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input["e_field"] = 4
        self.input["tip_height"] = 80
        self.input["tip_radius"] = 20
        self.input["z_height"] = 50
        self.input["tip_shank_angle"] = None
        self.input["basic_structure"] = None
        self.input["num_atoms"] = 5  # number of atoms to evaporate

    def create_input_structure(self):
        """Creates the tip structure"""
        tip_generator = TipGenerator(
            structure=self.input.basic_structure,
            h=self.input.tip_height,
            ah=self.input.tip_radius,
            alpha=self.input.tip_shank_angle,
            zheight=self.input.z_height,
        )
        self.input["structure"] = tip_generator.create_tip_pyiron(
            self.project
        )  # structure of the tip
        return tip_generator

    def run_static(self, **kwargs):
        tip_generator = self.create_input_structure()
        # self.input['structure']= tip_generator.create_tip_pyiron(self.project)
        job = RRModel(
            tip_generator=tip_generator,
            structure=self.input.structure,
            e_field=self.input.e_field,
        )
        job.run_evaporation(
            num_atoms=self.input.num_atoms, path=self.working_directory, **kwargs
        )
        self.collect_output()

        self.status.finished = True

    def collect_output(self, path=None):
        if path is None:
            path = self.working_directory

        fin_evapos = h5io.read_hdf5(f'{path}/fin_evapos.h5')
        tip_pos = h5io.read_hdf5(f'{path}/tip_pos.h5')
        tip_pos_charge = h5io.read_hdf5(f'{path}/tip_pos_charge.h5')
        tip_surf_ind = h5io.read_hdf5(f'{path}/tip_surf_ind.h5')
        self.output["evaporation_trajectories"] = fin_evapos
        self.output["tip_structures"] = tip_pos
        self.output["equilibrium_charges"] = tip_pos_charge
        self.output["surface_indices"] = tip_surf_ind
        self.to_hdf() #is a duplication of output to job hdf5 file


JobType.register(RRModelAPTjob)
