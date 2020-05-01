import unittest
import torch
import difftorch as dtorch


class APITestCase(unittest.TestCase):
    def rosenbrock(self, x):
        x, y = x[0], x[1]
        return (1. - x)**2 + 100. * (y - x**2)**2

    def rosenbrockGrad(self, x):
        x, y = x[0], x[1]
        return torch.tensor([-2*(1-x)-400*x*(-(x**2) + y), 200*(-(x**2) + y)])

    def rosenbrockHessian(self, x):
        x, y = x[0], x[1]
        return torch.tensor([[2.+1200.*x*x-400.*y, -400.*x], [-400.*x, 200.]])

    def fvect2vect2(self, x):
        x, y = x[0], x[1]
        return torch.stack([x*x*y, 5*x+y.sin()])

    def fvect2vect2Jacobian(self, x):
        x, y = x[0], x[1]
        return torch.tensor([[2*x*y, x*x], [5., y.cos()]])

    def fvect3vect4(self, x):
        y1, y2, y3, y4 = x[0], 5*x[2], 4*x[1]*x[1]-2*x[2], x[2]*x[0].sin()
        return torch.stack([y1, y2, y3, y4])

    def fvect3vect4Jacobian(self, x):
        return torch.tensor([[1, 0, 0], [0, 0, 5], [0, 8*x[1], -2], [x[2]*x[0].cos(), 0, x[0].sin()]])

    def fvect3vect3(self, x):
        r, theta, phi = x[0], x[1], x[2]
        return torch.stack([r*phi.sin()*theta.cos(), r*phi.sin()*theta.sin(), r*phi.cos()])

    def fvect3vect3Jacobian(self, x):
        r, theta, phi = x[0], x[1], x[2]
        return torch.tensor([[phi.sin()*theta.cos(), -r*phi.sin()*theta.sin(), r*phi.cos()*theta.cos()], [phi.sin()*theta.sin(), r*phi.sin()*theta.cos(), r*phi.cos()*theta.sin()], [phi.cos(), 0., -r*phi.sin()]])

    def test_g(self):
        x = torch.randn(2)
        g = dtorch.g(self.rosenbrock, x)
        gCorrect = self.rosenbrockGrad(x)
        self.assertTrue(gCorrect.allclose(g))

    def test_fg(self):
        x = torch.randn(2)
        z, g = dtorch.fg(self.rosenbrock, x)
        gCorrect = self.rosenbrockGrad(x)
        zCorrect = self.rosenbrock(x)
        self.assertTrue(gCorrect.allclose(g))
        self.assertTrue(zCorrect.allclose(z))

    def test_gvp(self):
        x = torch.randn(2)
        v = torch.randn(2)
        gv = dtorch.gvp(self.rosenbrock, x, v)
        gvCorrect = torch.dot(self.rosenbrockGrad(x), v)
        self.assertTrue(gvCorrect.allclose(gv))

    def test_fgvp(self):
        x = torch.randn(2)
        v = torch.randn(2)
        z, gv = dtorch.fgvp(self.rosenbrock, x, v)
        gvCorrect = torch.dot(self.rosenbrockGrad(x), v)
        zCorrect = self.rosenbrock(x)
        self.assertTrue(gvCorrect.allclose(gv))
        self.assertTrue(zCorrect.allclose(z))

    def test_j(self):
        x = torch.randn(2)
        j = dtorch.j(self.fvect2vect2, x)
        jCorrect = self.fvect2vect2Jacobian(x)
        self.assertTrue(jCorrect.allclose(j))

        x = torch.randn(3)
        j = dtorch.j(self.fvect3vect3, x)
        jCorrect = self.fvect3vect3Jacobian(x)
        self.assertTrue(jCorrect.allclose(j))

        x = torch.randn(3)
        j = dtorch.j(self.fvect3vect4, x)
        jCorrect = self.fvect3vect4Jacobian(x)
        self.assertTrue(jCorrect.allclose(j))

    def test_fj(self):
        x = torch.randn(3)
        z, j = dtorch.fj(self.fvect3vect4, x)
        jCorrect = self.fvect3vect4Jacobian(x)
        zCorrect = self.fvect3vect4(x)
        self.assertTrue(jCorrect.allclose(j))
        self.assertTrue(zCorrect.allclose(z))

    def test_jvp(self):
        x = torch.randn(2)
        v = torch.randn(2)
        jv = dtorch.jvp(self.fvect2vect2, x, v)
        jvCorrect = torch.matmul(self.fvect2vect2Jacobian(x), v)
        self.assertTrue(jvCorrect.allclose(jv))

        x = torch.randn(3)
        v = torch.randn(3)
        jv = dtorch.jvp(self.fvect3vect3, x, v)
        jvCorrect = torch.matmul(self.fvect3vect3Jacobian(x), v)
        self.assertTrue(jvCorrect.allclose(jv))

        x = torch.randn(3)
        v = torch.randn(3)
        jv = dtorch.jvp(self.fvect3vect4, x, v)
        jvCorrect = torch.matmul(self.fvect3vect4Jacobian(x), v)
        self.assertTrue(jvCorrect.allclose(jv))

    def test_fjvp(self):
        x = torch.randn(3)
        v = torch.randn(3)
        z, jv = dtorch.fjvp(self.fvect3vect4, x, v)
        jvCorrect = torch.matmul(self.fvect3vect4Jacobian(x), v)
        zCorrect = self.fvect3vect4(x)
        self.assertTrue(jvCorrect.allclose(jv))
        self.assertTrue(zCorrect.allclose(z))

    def test_vjp(self):
        x = torch.randn(2)
        v = torch.randn(2)
        jv = dtorch.vjp(self.fvect2vect2, x, v)
        jvCorrect = torch.matmul(v, self.fvect2vect2Jacobian(x))
        self.assertTrue(jvCorrect.allclose(jv))

        x = torch.randn(3)
        v = torch.randn(3)
        jv = dtorch.vjp(self.fvect3vect3, x, v)
        jvCorrect = torch.matmul(v, self.fvect3vect3Jacobian(x))
        self.assertTrue(jvCorrect.allclose(jv))

        x = torch.randn(3)
        v = torch.randn(4)
        jv = dtorch.vjp(self.fvect3vect4, x, v)
        jvCorrect = torch.matmul(v, self.fvect3vect4Jacobian(x))
        self.assertTrue(jvCorrect.allclose(jv))

    def test_fvjp(self):
        x = torch.randn(3)
        v = torch.randn(4)
        z, jv = dtorch.fvjp(self.fvect3vect4, x, v)
        jvCorrect = torch.matmul(v, self.fvect3vect4Jacobian(x))
        zCorrect = self.fvect3vect4(x)
        self.assertTrue(jvCorrect.allclose(jv))
        self.assertTrue(zCorrect.allclose(z))

    def test_h(self):
        x = torch.randn(2)
        h = dtorch.h(self.rosenbrock, x)
        hCorrect = self.rosenbrockHessian(x)
        self.assertTrue(hCorrect.allclose(h))

    def test_fh(self):
        x = torch.randn(2)
        z, h = dtorch.fh(self.rosenbrock, x)
        hCorrect = self.rosenbrockHessian(x)
        zCorrect = self.rosenbrock(x)
        self.assertTrue(hCorrect.allclose(h))
        self.assertTrue(zCorrect.allclose(z))

    def test_hvp(self):
        x = torch.randn(2)
        v = torch.randn(2)
        hv = dtorch.hvp(self.rosenbrock, x, v)
        hvCorrect = torch.matmul(self.rosenbrockHessian(x), v)
        self.assertTrue(hvCorrect.allclose(hv))

    def test_fhvp(self):
        x = torch.randn(2)
        v = torch.randn(2)
        z, hv = dtorch.fhvp(self.rosenbrock, x, v)
        hvCorrect = torch.matmul(self.rosenbrockHessian(x), v)
        zCorrect = self.rosenbrock(x)
        self.assertTrue(hvCorrect.allclose(hv))
        self.assertTrue(zCorrect.allclose(z))
