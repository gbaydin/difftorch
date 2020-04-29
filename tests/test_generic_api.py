import unittest
import torch
import difftorch


class GenericAPITestCase(unittest.TestCase):
    def f(self, x, y):
        return torch.nn.functional.conv3d(x, y, padding=2, stride=2)

    def f2(self, x, y):
        return x[0, 0, 0, 0, 0] + y[0, 0, 0, 0, 0], torch.nn.functional.conv3d(x, y, padding=2, stride=2)

    def test_generic_jacobianv(self):
        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        xd = x.clone()
        yd = y.clone()
        jv = difftorch.generic_jacobianv(self.f, (x, y), (xd, yd))
        self.assertTrue(True)

        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        xd = x.clone()
        yd = y.clone()
        jv = difftorch.generic_jacobianv(self.f2, (x, y), (xd, yd))
        self.assertTrue(True)

    def test_generic_jacobianTv(self):
        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z = self.f(x, y)
        zd = z.clone()
        jtvx, jtvy = difftorch.generic_jacobianTv(self.f, (x, y), zd)
        self.assertTrue(True)

        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z1, z2 = self.f2(x, y)
        z1d = z1.clone()
        z2d = z2.clone()
        jtvx, jtvy = difftorch.generic_jacobianTv(self.f2, (x, y), (z1d, z2d))
        self.assertTrue(True)
