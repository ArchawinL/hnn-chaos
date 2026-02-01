import torch
import numpy as np
from scipy.integrate import solve_ivp 


def get_dataset(system, n_samples=100, t_span=[0, 1]):
    

    X, y = [], []
    
    def dynamics(t, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Get field from state_tensor
        dstate = system.get_derivatives(None, state_tensor)
        return dstate.squeeze().detach().numpy()

    for _ in range(n_samples):
        # TODO: Random initialization of q and p
        y0 = torch.rand_like()
        pass
        
    # Return dataset tensors
    return torch.tensor(X), torch.tensor(y)