import torch
import torch.nn as nn

class QuasimotoWave(nn.Module):
    """
    Author: QueenFi703
    Description: Learnable continuous latent wave representation with 
    controlled phase irregularity and localized Gaussian envelopes.
    """
    def __init__(self):
        super().__init__()
        # Core parameters
        self.A = nn.Parameter(torch.tensor(1.0))
        self.k = nn.Parameter(torch.randn(()))
        self.omega = nn.Parameter(torch.randn(()))
        self.v = nn.Parameter(torch.randn(()))
        self.log_sigma = nn.Parameter(torch.zeros(()))
        self.phi = nn.Parameter(torch.zeros(()))

        # Irregularity controls
        self.epsilon = nn.Parameter(torch.tensor(0.1))
        self.lmbda = nn.Parameter(torch.randn(()))

    def forward(self, x, t):
        sigma = torch.exp(self.log_sigma)
        
        # Plane-wave phase
        phase = self.k * x - self.omega * t
        
        # Gaussian envelope (locality bias)
        envelope = torch.exp(-0.5 * ((x - self.v * t) / sigma) ** 2)
        
        # QueenFi703 Phase Distortion
        modulation = torch.sin(self.phi + self.epsilon * torch.cos(self.lmbda * x))
        
        # Quasimoto State (Complex)
        return self.A * torch.exp(1j * phase) * envelope * modulation

class QuasimotoLayer(nn.Module):
    """
    Feature layer for deep networks using an ensemble of Quasimoto waves.
    """
    def __init__(self, out_features):
        super().__init__()
        self.waves = nn.ModuleList([QuasimotoWave() for _ in range(out_features)])

    def forward(self, x, t):
        psi_stack = torch.stack([w(x, t) for w in self.waves], dim=-1)
        # Concatenate real and imaginary parts for MLP compatibility
        return torch.cat([psi_stack.real, psi_stack.imag], dim=-1)