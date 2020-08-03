import numpy as np
import torch
import matplotlib.pyplot as plt
class summ(torch.nn.Module):
    def __init__(
        self,
    ):
        super(summ,self).__init__()
        pass
    def forward(
        self,
        a,
    ):
        pass
        ans = torch.zeros_like(a)
        for i in range(1,a.shape[0]):
            ans[i] = ans[i-1] + a[i-1]
        return ans
class fitter:
    def __init__(
        self,
        pose
    ):
        pass

    def step(self):
        pass
    pass
if __name__=="__main__":
    a = torch.ones((6,3))
    av = torch.autograd.Variable(torch.rand((6,3)),requires_grad=True)
    SUM = summ()
    print(SUM(av))
    
    plotter = []

    adam = torch.optim.Adam([av])
    for i in range(50000):
        par = SUM(av)-SUM(a)
        LOSS = par[i//10000+1].abs().mean()
        LOSS.backward()
        av.grad = av.grad * torch.eye(6)[i//10000].unsqueeze(1)
        adam.step()
        if (i%10==0):
            print(par,LOSS)
            plotter.append(float(LOSS.mean()))
    plt.plot(plotter)
    plt.savefig("plot01.png")
    pass