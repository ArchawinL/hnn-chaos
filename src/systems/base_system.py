import torch 
from abc import ABC, abstractmethod


class BaseSystem(ABC):
     
    def __init__(self, device='cpu'):
        self.device = device


    @abstractmethod
    def get_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:

        pass

    
    def get_derivatives(self, t, state):
        
        state = state.detach().clone().requires_grad_(True)

        # State needs position q and momenta p
        # TODO Check this line, but should be split evenly
        dim = state.shape[1] // 2

        q, p = state[:, :dim], state[:, dim:]

        # Compute H, why sum?
        H = self.get_hamiltonian(q, p).sum()

        grads = torch.autograd.grad(H, [q, p], create_graph=True)
        dH_dq, dH_dp = grads[0], grads[1]


        # Return Hamiltonian
        return torch.cat([dH_dp, -dH_dq], dim=1)
        
