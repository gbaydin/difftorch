import unittest
import torch
import difftorch


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

    def test_grad(self):
        x = torch.randn(2)
        g = difftorch.grad(self.rosenbrock, x)
        gCorrect = self.rosenbrockGrad(x)
        self.assertTrue(gCorrect.allclose(g))

    def test_gradv(self):
        x = torch.randn(2)
        v = torch.randn(2)
        gv = difftorch.gradv(self.rosenbrock, x, v)
        gvCorrect = torch.dot(self.rosenbrockGrad(x), v)
        self.assertTrue(gvCorrect.allclose(gv))

    def test_diff(self):
        x = torch.randn(1)[0]
        d = difftorch.diff(torch.sin, x)
        dCorrect = torch.cos(x)
        self.assertTrue(dCorrect.allclose(d))

    def test_jacobian(self):
        x = torch.randn(2)
        j = difftorch.jacobian(self.fvect2vect2, x)
        jCorrect = self.fvect2vect2Jacobian(x)
        self.assertTrue(jCorrect.allclose(j))

        x = torch.randn(3)
        j = difftorch.jacobian(self.fvect3vect3, x)
        jCorrect = self.fvect3vect3Jacobian(x)
        self.assertTrue(jCorrect.allclose(j))

        x = torch.randn(3)
        j = difftorch.jacobian(self.fvect3vect4, x)
        jCorrect = self.fvect3vect4Jacobian(x)
        self.assertTrue(jCorrect.allclose(j))

    def test_jacobianv(self):
        x = torch.randn(2)
        v = torch.randn(2)
        jv = difftorch.jacobianv(self.fvect2vect2, x, v)
        jvCorrect = torch.matmul(self.fvect2vect2Jacobian(x), v)
        self.assertTrue(jvCorrect.allclose(jv))

        x = torch.randn(3)
        v = torch.randn(3)
        jv = difftorch.jacobianv(self.fvect3vect3, x, v)
        jvCorrect = torch.matmul(self.fvect3vect3Jacobian(x), v)
        self.assertTrue(jvCorrect.allclose(jv))

        x = torch.randn(3)
        v = torch.randn(3)
        jv = difftorch.jacobianv(self.fvect3vect4, x, v)
        jvCorrect = torch.matmul(self.fvect3vect4Jacobian(x), v)
        self.assertTrue(jvCorrect.allclose(jv))

    def test_jacobianTv(self):
        x = torch.randn(2)
        v = torch.randn(2)
        jv = difftorch.jacobianTv(self.fvect2vect2, x, v)
        jvCorrect = torch.matmul(v, self.fvect2vect2Jacobian(x))
        self.assertTrue(jvCorrect.allclose(jv))

        x = torch.randn(3)
        v = torch.randn(3)
        jv = difftorch.jacobianTv(self.fvect3vect3, x, v)
        jvCorrect = torch.matmul(v, self.fvect3vect3Jacobian(x))
        self.assertTrue(jvCorrect.allclose(jv))

        x = torch.randn(3)
        v = torch.randn(4)
        jv = difftorch.jacobianTv(self.fvect3vect4, x, v)
        jvCorrect = torch.matmul(v, self.fvect3vect4Jacobian(x))
        self.assertTrue(jvCorrect.allclose(jv))

    def test_hessian(self):
        x = torch.randn(2)
        h = difftorch.hessian(self.rosenbrock, x)
        hCorrect = self.rosenbrockHessian(x)
        self.assertTrue(hCorrect.allclose(h))

    def test_hessianv(self):
        x = torch.randn(2)
        v = torch.randn(2)
        hv = difftorch.hessianv(self.rosenbrock, x, v)
        hvCorrect = torch.matmul(self.rosenbrockHessian(x), v)
        self.assertTrue(hvCorrect.allclose(hv))

    def test_laplacian(self):
        x = torch.randn(2)
        ll = difftorch.laplacian(self.rosenbrock, x)
        lCorrect = self.rosenbrockHessian(x).trace()
        self.assertTrue(lCorrect.allclose(ll))

    def test_curl(self):
        x = torch.tensor([1.5, 2.5, 0.2])
        c = difftorch.curl(self.fvect3vect3, x)
        cCorrect = torch.tensor([-0.879814, -2.157828, 0.297245])
        self.assertTrue(cCorrect.allclose(c))

    def test_div(self):
        x = torch.tensor([1.5, 2.5, 0.2])
        d = difftorch.div(self.fvect3vect3, x)
        dCorrect = torch.tensor(-0.695911)
        self.assertTrue(dCorrect.allclose(d))
