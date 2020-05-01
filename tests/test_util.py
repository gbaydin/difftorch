import unittest
import torch
from difftorch import util


class UtilTestCase(unittest.TestCase):
    def f(self, x, y):
        return torch.nn.functional.conv3d(x, y, padding=2, stride=2)

    def test_flatten_unflatten(self):
        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        f = util.flatten((x, y))
        f2 = util.flatten(util.unflatten_as(f, (x, y)))
        self.assertTrue(f.allclose(f2))

    def test_unzip(self):
        lst = [(1, 2), (3, 4), (8, 9)]
        l1, l2 = util.unzip(lst)
        l1Correct = (1, 3, 8)
        l2Correct = (2, 4, 9)
        self.assertEqual(l1Correct, l1)
        self.assertEqual(l2Correct, l2)
