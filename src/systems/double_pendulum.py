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

        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        
        V = -m1 * g * l1 * torch.cos(theta1) - m2 * g * (l1 * torch.cos(theta1) + l2 * torch.cos(theta2))

        # Need analytical solution
        
        # Calculate determinant
        delta = theta1 - theta2
        det_M = l1**2 * l2**2 * m2 * (m1 + m2 * torch.sin(delta)**2)
        
        # Calculate terms for kinetic
        term1 = (m2 * l2**2) * p1**2
        term2 = ((m1 + m2) * l1**2) * p2**2
        term3 = -2 * (m2 * l1 * l2 * torch.cos(delta)) * p1 * p2

        T = 0.5 * (term1 + term2 + term3) / (det_M + 1e-8) #  Epsilon prevents gradient explosion

        return T + V