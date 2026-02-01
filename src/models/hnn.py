import torch 
import torch.nn as nn




class HNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=200):
        super.__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        

    def forward(self, x):
        
        return self.net(x)
    

    def time_derivative(self, x):

        x = x.requries_grad_(True)
        H = self.forward(x).sum()

        dH_dx = torch.autogra.grad(H, x, create_graph = True)[0]

        input_dim = x.shape[1]
        half_dim = input_dim // 2
        
        dH_dq = dH_dx[:, :half_dim] 
        dH_dp = dH_dx[:, half_dim:]


        field = torch.cat([dH_dp, -dH_dq], dim=1)


        return field