import torch
from .base_system import BaseSystem

class DoublePendulum(BaseSystem):

    # Tracking two masses and two lengths
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81, device='cpu'):
        super().__init__(device)
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g

    def get_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Calculates Total Energy H = T + V
        q: [theta1, theta2]
        p: [p1, p2]
        """
        theta1, theta2 = q[:, 0], q[:, 1]
        p1, p2 = p[:, 0], p[:, 1]
        
        V = -self.m1 * self.g * self.l1 * torch.cos(theta1) - self.m2 * self.g * (self.l1 * torch.cos(theta1) + self.l2 * torch.cos(theta2))

        # TODO: Implement the Kinetic Energy (T) equation
        T = torch.zeros_like(p1) # Placeholder

        return T + V