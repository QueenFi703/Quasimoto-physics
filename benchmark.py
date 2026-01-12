import torch
import torch.nn as nn
import torch.optim as optim
from quasimoto import QuasimotoWave

def run_benchmark():
    print("Initializing Quasimoto Research Benchmark...")
    
    # Task: Fit a non-stationary signal (Chirp with a glitch)
    x = torch.linspace(-10, 10, 1000)
    t = torch.zeros_like(x)
    y_target = torch.sin(0.5 * x**2) * torch.exp(-0.1 * x**2)
    
    # Quasimoto Ensemble
    model = nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    # For this proof-of-concept, we optimize a single Quasimoto wave
    q_wave = QuasimotoWave()
    optimizer = optim.Adam(q_wave.parameters(), lr=0.01)
    
    for i in range(1000):
        optimizer.zero_grad()
        psi = q_wave(x, t)
        loss = torch.mean((psi.real - y_target)**2)
        loss.backward()
        optimizer.step()
        
        if i % 200 == 0:
            print(f"Iteration {i} | Loss: {loss.item():.6f}")

    print("Benchmark Complete. Credits: QueenFi703")

if __name__ == "__main__":
    run_benchmark()