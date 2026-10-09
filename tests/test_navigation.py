import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('participant', ROOT / 'controllers/participant/participant.py')
nav = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nav)


class Robot:
    def __init__(self):
        self.p = np.array([0., 0., .3])
        self.command = np.zeros(3)

    def lidar_info(self):
        return 2*np.pi, 360, 12., 24, 1.2

    def camera_info(self):
        return 320, 240, 1.2

    def position(self):
        return self.p

    def imu(self):
        return 0., 0., 0.

    def time(self):
        return 2.

    def drive(self, vx=0., vy=0., vyaw=0.):
        self.command = np.array([vx, vy, vyaw])

    def stop(self):
        self.command[:] = 0

    stand = stop


def observation(colour, x, y=0.):
    return colour, np.array([float(x), float(y)]), 2.


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.robot = Robot()
        self.navigator = nav.Navigator(self.robot)

    def observe(self, observations, now):
        return self.navigator.observe_markers(observations, self.robot.p, now)

    def test_referee_zones_match_visible_pads(self):
        world = (ROOT / 'worlds/level1.wbt').read_text(encoding='utf-8')
        config = json.loads((ROOT / 'config/level1.json').read_text(encoding='utf-8'))
        for pad in config['checkpoints'] + [config['finish']]:
            pattern = r'DEF\s+' + re.escape(pad['def']) + r'\s+Checkpoint\s*\{\s*translation\s+(\S+)\s+(\S+)'
            match = re.search(pattern, world)
            self.assertIsNotNone(match, pad['name'])
            np.testing.assert_allclose([pad['x'], pad['y']], [float(v) for v in match.groups()], atol=1e-8)

    def test_reaching_yellow_pad_does_not_confirm_it(self):
        for now in (0., .1, .2, .3):
            self.assertEqual(self.observe([observation('yellow', 0)], now), 0)
        self.assertFalse(self.navigator.target['visited'])

    def test_next_yellow_requires_repeated_current_observations(self):
        self.observe([observation('yellow', 0)], 0.)
        self.assertEqual(self.observe([observation('yellow', 2)], 1.), 0)
        self.assertEqual(self.observe([observation('yellow', 2)], 1.1), 1)
        np.testing.assert_allclose(self.navigator.target['xy'], [2., 0.])

    def test_still_visible_yellow_prevents_premature_transition(self):
        for now in (0., .1, .2, .3):
            self.assertEqual(self.observe([observation('yellow', 0), observation('yellow', 2)], now), 0)
        np.testing.assert_allclose(self.navigator.target['xy'], [0., 0.])

    def test_old_or_distant_yellow_does_not_confirm_current_target(self):
        self.observe([observation('yellow', 2)], 0.)
        self.observe([observation('yellow', 4)], 1.)
        self.assertEqual(self.observe([observation('yellow', 4)], 1.1), 0)
        self.robot.p[0] = 2.
        self.assertEqual(self.observe([], 5.), 0)

    def test_last_pad_is_verified_before_finish(self):
        for index in range(3):
            self.observe([observation('green', -3-index*2)], 0.)
        self.observe([observation('yellow', 0), observation('purple', 3)], 0.)
        self.navigator.sensors.scan = lambda *args: (np.empty((0, 2)), np.empty((0, 2)))
        self.navigator.sensors.markers = lambda *args: []
        with contextlib.redirect_stdout(io.StringIO()):
            self.navigator.tick(self.robot)
        self.assertEqual(sum(m['visited'] for m in self.navigator.markers), 3)
        self.assertLess(self.robot.command[0], 0)
        self.assertIsNotNone(self.navigator.verify_point)
        self.assertEqual(self.observe([observation('green', 0)], 3.), 4)
        self.assertEqual(self.navigator.target['colour'], 'purple')
        self.assertIsNone(self.navigator.verify_point)

    def test_converged_tracks_count_as_one_checkpoint(self):
        self.observe([observation('green', 2), observation('green', 3)], 0.)
        self.navigator.markers[1]['xy'] = np.array([2.3, 0.])
        self.assertEqual(self.observe([], 1.), 1)

    def test_new_free_scan_clears_old_wall_but_not_occluded_cells(self):
        grid = nav.CourseMap()
        wall = np.array([[1., 0.], [2., 0.]])
        empty = np.empty((0, 2))
        grid.update(wall, empty, self.robot.p, .1)
        old_cell, hidden_cell = [tuple(np.floor(p/nav.CELL).astype(int)) for p in wall]
        grid.update(empty, wall[:1], self.robot.p, .1)
        self.assertNotIn(old_cell, grid.walls)
        self.assertIn(hidden_cell, grid.walls)

    def test_current_wall_hit_wins_over_free_ray(self):
        grid = nav.CourseMap()
        wall = np.array([[1., 0.]])
        grid.update(wall, wall, self.robot.p, .1)
        cell = tuple(np.floor(wall[0]/nav.CELL).astype(int))
        self.assertIn(cell, grid.walls)
        self.assertNotIn(cell, grid.free)


if __name__ == '__main__':
    unittest.main()
