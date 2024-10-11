# unittest test cases for the main module of the RobinRollandModel package
import unittest
import numpy as np
from RobinRollandModel.main import RRModel

# test_main.py

class TestRRModel(unittest.TestCase):

    def setUp(self):
        # Mock structure with positions
        self.structure = type('MockStructure', (object,), {
            'positions': np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
        })()

        # Surface indices
        self.surf_indices = np.array([0, 1, 2, 3])

        # Charge distribution
        self.new_charge = np.array([1.0, 1.0, 1.0, 1.0])

        # Evaporating atom index
        self.evap_ind = 0

    def test_evaporation_trajectory_force_cut_output_type(self):
        trajectory = RRModel.evaporation_trajectory_force_cut(
            self.structure, self.surf_indices, self.new_charge, self.evap_ind
        )
        self.assertIsInstance(trajectory, np.ndarray)

    def test_evaporation_trajectory_force_cut_output_shape(self):
        trajectory = RRModel.evaporation_trajectory_force_cut(
            self.structure, self.surf_indices, self.new_charge, self.evap_ind
        )
        self.assertEqual(trajectory.shape[1], 3)

    def test_evaporation_trajectory_force_cut_known_input(self):
        expected_trajectory = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]  # Assuming force cutoff is reached immediately
        ])
        trajectory = RRModel.evaporation_trajectory_force_cut(
            self.structure, self.surf_indices, self.new_charge, self.evap_ind, force_cutoff=1e10
        )
        np.testing.assert_array_almost_equal(trajectory, expected_trajectory)

    def test_evaporation_trajectory_force_cut_edge_case(self):
        trajectory = RRModel.evaporation_trajectory_force_cut(
            self.structure, self.surf_indices, self.new_charge, self.evap_ind, force_cutoff=1e-12
        )
        self.assertGreater(len(trajectory), 1)

    def test_evaporation_trajectory_force_cut_no_movement(self):
        trajectory = RRModel.evaporation_trajectory_force_cut(
            self.structure, self.surf_indices, self.new_charge, self.evap_ind, force_cutoff=0
        )
        expected_trajectory = np.array([
            [0.0, 0.0, 0.0]
        ])
        np.testing.assert_array_almost_equal(trajectory, expected_trajectory)

    def test_evaporation_trajectory_force_cut_large_time_step(self):
        trajectory = RRModel.evaporation_trajectory_force_cut(
            self.structure, self.surf_indices, self.new_charge, self.evap_ind, dt=10
        )
        self.assertGreater(len(trajectory), 1)

if __name__ == '__main__':
    unittest.main()

    def test_evaporation_trajectory_force_cut_known_input(self):
        expected_trajectory = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]  # Assuming force cutoff is reached immediately
        ])
        trajectory = RRModel.evaporation_trajectory_force_cut(
            self.structure, self.surf_indices, self.new_charge, self.evap_ind, force_cutoff=1e10
        )
        np.testing.assert_array_almost_equal(trajectory, expected_trajectory)

    def test_evaporation_trajectory_force_cut_edge_case(self):
        trajectory = RRModel.evaporation_trajectory_force_cut(
            self.structure, self.surf_indices, self.new_charge, self.evap_ind, force_cutoff=1e-12
        )
        self.assertGreater(len(trajectory), 1)

if __name__ == '__main__':
    unittest.main()