import unittest
import numpy as np
from lnwpca.geometry import rigid_fit, transform, refine


class GeometryTests(unittest.TestCase):
    def test_known_transform(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(100, 3))
        angle = 0.4
        truth = np.eye(4)
        truth[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        truth[:3, 3] = [.1, -.2, .3]
        target = transform(source, truth)
        np.testing.assert_allclose(rigid_fit(source, target), truth, atol=1e-12)

    def test_zero_weight_outlier(self):
        source = np.random.default_rng(3).normal(size=(20, 3))
        target = source + .01
        target[-1] = 100
        weights = np.ones(20)
        weights[-1] = 0
        matrix = rigid_fit(source, target, weights)
        np.testing.assert_allclose(matrix[:3, 3], .01, atol=1e-12)

    def test_refinement(self):
        source = np.random.default_rng(8).normal(size=(50, 3))
        target = source + .001
        matrix, history = refine(source, target)
        np.testing.assert_allclose(transform(source, matrix), target, atol=1e-10)
        self.assertLess(history[-1], 1e-10)


if __name__ == '__main__':
    unittest.main()
