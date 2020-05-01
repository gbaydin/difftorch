import unittest
import torch
import difftorch as dtorch


class GenericAPITestCase(unittest.TestCase):
    def f(self, x, y):
        return torch.nn.functional.conv3d(x, y, padding=2, stride=2)

    def f2(self, x, y):
        return x[0, 0, 0, 0, 0] + y[0, 0, 0, 0, 0], torch.nn.functional.conv3d(x, y, padding=2, stride=2)

    def test_generic_fjacobianv(self):
        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z = self.f(x, y)
        xd = x.clone()
        yd = y.clone()
        zz, jv = dtorch.generic_fjacobianv(self.f, (x, y), (xd, yd))
        self.assertTrue(z.shape == jv.shape)
        self.assertTrue(z.allclose(zz))

        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z1, z2 = self.f2(x, y)
        xd = x.clone()
        yd = y.clone()
        (zz1, zz2), (jv1, jv2) = dtorch.generic_fjacobianv(self.f2, (x, y), (xd, yd))
        self.assertTrue(z1.shape == jv1.shape)
        self.assertTrue(z2.shape == jv2.shape)
        self.assertTrue(z1.allclose(zz1))
        self.assertTrue(z2.allclose(zz2))

    def test_generic_jacobianv(self):
        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z = self.f(x, y)
        xd = x.clone()
        yd = y.clone()
        jv = dtorch.generic_jacobianv(self.f, (x, y), (xd, yd))
        self.assertTrue(z.shape == jv.shape)

        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z1, z2 = self.f2(x, y)
        xd = x.clone()
        yd = y.clone()
        jv1, jv2 = dtorch.generic_jacobianv(self.f2, (x, y), (xd, yd))
        self.assertTrue(z1.shape == jv1.shape)
        self.assertTrue(z2.shape == jv2.shape)

    def test_generic_fjacobianTv(self):
        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z = self.f(x, y)
        zd = z.clone()
        zz, (jtvx, jtvy) = dtorch.generic_fjacobianTv(self.f, (x, y), zd)
        self.assertTrue(x.shape == jtvx.shape)
        self.assertTrue(y.shape == jtvy.shape)
        self.assertTrue(z.allclose(zz))

        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z1, z2 = self.f2(x, y)
        z1d = z1.clone()
        z2d = z2.clone()
        (zz1, zz2), (jtvx, jtvy) = dtorch.generic_fjacobianTv(self.f2, (x, y), (z1d, z2d))
        self.assertTrue(x.shape == jtvx.shape)
        self.assertTrue(y.shape == jtvy.shape)
        self.assertTrue(z1.allclose(zz1))
        self.assertTrue(z2.allclose(zz2))

    def test_generic_jacobianTv(self):
        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z = self.f(x, y)
        zd = z.clone()
        jtvx, jtvy = dtorch.generic_jacobianTv(self.f, (x, y), zd)
        self.assertTrue(x.shape == jtvx.shape)
        self.assertTrue(y.shape == jtvy.shape)

        x = torch.randn([16, 2, 4, 4, 4])
        y = torch.randn([3, 2, 3, 3, 3])
        z1, z2 = self.f2(x, y)
        z1d = z1.clone()
        z2d = z2.clone()
        jtvx, jtvy = dtorch.generic_jacobianTv(self.f2, (x, y), (z1d, z2d))
        self.assertTrue(x.shape == jtvx.shape)
        self.assertTrue(y.shape == jtvy.shape)
