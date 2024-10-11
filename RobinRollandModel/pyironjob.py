"""
Module for pyiron job based on the RRModel module
"""

import os
import logging
from pyiron_base.utils.error import ImportAlarm
from pyiron_base.jobs.job.template import TemplateJob
from pyiron_base.jobs.job.jobtype import JobType
import h5io

# Configure logging
# Configure logging
logging.basicConfig(
    filename="simulation_pyiron.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

try:
    from RobinRollandModel.updated_main import RRModel, SimulationConfig
    from RobinRollandModel.datautils import TipGenerator
except ImportError as e:
    import_alarm = ImportAlarm("Unable to import rrmodel or datautils")
    import_alarm()


class RRModelAPTjob(TemplateJob):
    """Pyiron job class for RRModelAPT simulation"""

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        # Initialize default input parameters
        self.input["e_field"] = 4.0
        self.input["tip_height"] = 80.0
        self.input["tip_radius"] = 20.0
        self.input["z_height"] = 50.0
        self.input["tip_shank_angle"] = None
        self.input["basic_structure"] = None  # User must provide this
        self.input["num_atoms"] = 5  # Number of atoms to evaporate
        self.input["simulation_steps"] = 1000
        self.input["epsilon"] = 1e-9
        self.input["dt"] = 1.5
        self.input["num_traj_steps"] = 200

    def create_input_structure(self):
        """
        Creates the tip structure using the TipGenerator.
        
        Returns
        -------
        TipGenerator
            An instance of the TipGenerator used to create the tip structure.
        """
        # Validate input
        if self.input["basic_structure"] is None:
            raise ValueError("basic_structure must be provided in the input.")
        if not hasattr(self.input["basic_structure"], 'positions'):
            raise TypeError("basic_structure must be an Atoms object with 'positions' attribute.")

        # Create the tip structure
        tip_generator = TipGenerator(
            structure=self.input["basic_structure"],
            h=self.input["tip_height"],
            ah=self.input["tip_radius"],
            alpha=self.input["tip_shank_angle"],
            zheight=self.input["z_height"],
        )
        self.input["structure"] = tip_generator.create_tip_pyiron(
            self.project
        )  # Structure of the tip
        logging.info("Tip structure created successfully.")
        return tip_generator

    def run_static(self, **kwargs):
        """
        Runs the RRModel evaporation simulation.
        
        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments to configure the simulation.
        """
        # Create input structure
        tip_generator = self.create_input_structure()

        # Initialize the RRModel
        job = RRModel(
            tip_generator=tip_generator,
            structure=self.input["structure"],
            e_field=self.input["e_field"],
        )

        # Combine input parameters and kwargs into SimulationConfig
        sim_config = SimulationConfig(
            e_field=self.input["e_field"],
            radius=self.input["tip_radius"],
            steps=self.input["simulation_steps"],
            epsilon=self.input["epsilon"],
            zheight=self.input["z_height"]
            # Add other parameters as needed
        )

        # Run evaporation simulation
        logging.info("Starting evaporation simulation...")
        job.run_evaporation(
            num_atoms=self.input["num_atoms"],
            path=self.working_directory,
            config=sim_config,
            dt=self.input["dt"],
            num_steps=self.input["num_traj_steps"],
            **kwargs
        )

        # Collect the output
        self.collect_output()

        self.status.finished = True
        logging.info("Evaporation simulation completed.")

    def collect_output(self, path=None):
        """
        Collects output data from the RRModel simulation.
        
        Parameters
        ----------
        path : str, optional
            The path to the working directory containing the output files.
        """
        if path is None:
            path = self.working_directory

        try:
            fin_evapos = h5io.read_hdf5(os.path.join(path, 'fin_evapos.h5'))
            tip_pos = h5io.read_hdf5(os.path.join(path, 'tip_pos.h5'))
            tip_pos_charge = h5io.read_hdf5(os.path.join(path, 'tip_pos_charge.h5'))
            tip_surf_ind = h5io.read_hdf5(os.path.join(path, 'tip_surf_ind.h5'))
            det_coordinates = h5io.read_hdf5(os.path.join(path, 'det_coordinates.h5'))

            self.output["evaporation_trajectories"] = fin_evapos
            self.output["tip_structures"] = tip_pos
            self.output["equilibrium_charges"] = tip_pos_charge
            self.output["surface_indices"] = tip_surf_ind
            self.output["detector_coordinates"] = det_coordinates

            # Save output to job hdf5 file
            self.to_hdf()
            logging.info("Output data collected and saved successfully.")
        except Exception as e:
            logging.error("Failed to read output files: %s", e)
            raise

    def to_hdf(self, hdf=None, group_name=None):
        """
        Writes the job data to HDF5 format.
        
        Parameters
        ----------
        hdf : GenericHDF5, optional
            The HDF5 group to write to.
        group_name : str, optional
            The name of the group within the HDF5 file.
        """
        super().to_hdf(hdf=hdf, group_name=group_name)
        # Ensure that output data is stored in the HDF5 file
        if hdf is None:
            hdf = self.project_hdf5
        with hdf.open("output") as hdf_output:
            for key, value in self.output.items():
                hdf_output[key] = value
        logging.info("Job data written to HDF5 file.")

    @property
    def structure(self):
        """
        Returns the structure associated with the job.
        
        Returns
        -------
        Atoms
            The structure of the tip.
        """
        return self.input["structure"]


JobType.register(RRModelAPTjob)
