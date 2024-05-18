import torch

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is not None:
                    self.velocity[i] = self.momentum * self.velocity[i] + self.lr * param.grad
                    param -= self.velocity[i]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class RMSprop:
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.sq_grads = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is not None:
                    self.sq_grads[i] = self.alpha * self.sq_grads[i] + (1 - self.alpha) * param.grad**2
                    param -= self.lr * param.grad / (self.sq_grads[i].sqrt() + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        
        for i in range(len(self.params)):
            if self.params[i].grad is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.params[i].grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * self.params[i].grad**2
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                self.params[i].data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
