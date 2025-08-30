import torch
import torch.nn as nn
import torch.optim as optim

from verskyt.layers.projection import TverskyProjectionLayer

# Debug the parameter learning issue
layer = TverskyProjectionLayer(
    in_features=3,
    num_prototypes=2,
    num_features=2,
    learnable_ab=True,
    alpha=0.5,
    beta=0.5,
)

print(f"Initial alpha: {layer.alpha}")
print(f"Initial beta: {layer.beta}")

# Training step
x = torch.randn(4, 3)
targets = torch.randint(0, 2, (4,))
optimizer = optim.SGD(layer.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# Store initial values
initial_alpha = layer.alpha.clone().detach()
initial_beta = layer.beta.clone().detach()

optimizer.zero_grad()
output = layer(x)
loss = criterion(output, targets)
loss.backward()

print(f"\nAfter backward:")
print(f"Alpha grad: {layer.alpha.grad}")
print(f"Beta grad: {layer.beta.grad}")

optimizer.step()

print(f"\nAfter optimizer step:")
print(f"Alpha: {layer.alpha} (changed: {not torch.equal(layer.alpha, initial_alpha)})")
print(f"Beta: {layer.beta} (changed: {not torch.equal(layer.beta, initial_beta)})")

# Check if gradients are zero
if layer.alpha.grad is not None:
    print(f"Alpha grad magnitude: {layer.alpha.grad.abs().item()}")
if layer.beta.grad is not None:
    print(f"Beta grad magnitude: {layer.beta.grad.abs().item()}")
