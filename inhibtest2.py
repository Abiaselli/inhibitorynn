import torch
import torch.nn as nn
import torch.optim as optim

class InhibitoryNeuronLayer(nn.Module):
    def __init__(self, size, device='cpu'):
        super().__init__()
        self.device = device
        self.inhibition = nn.Parameter(torch.randn(size, dtype=torch.float32, device=device))  # start with diversity
        self.reward_lr = 11
        self.punishment_lr = 10

    def forward(self, input_signals):
        activations = (1.0 - torch.sigmoid(self.inhibition)) * input_signals
        return activations

    def reinforce(self, activations, reward_signal):
        with torch.no_grad():
            if reward_signal > 0:
                self.inhibition -= self.reward_lr * activations
            else:
                self.inhibition += self.punishment_lr * activations
            self.inhibition.clamp_(-10.0, 10.0)

class InhibitorySelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, inhibition_scores):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        #print(f"attn_scores shape: {attn_scores.shape}")
        #print(f"inhibition_scores shape: {inhibition_scores.shape}")
        #print(f"inhibition_scores unsqueezed shape: {inhibition_scores.unsqueeze(1).shape}")

        attn_scores = attn_scores - inhibition_scores.unsqueeze(1)  # inhibitory mask
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)

class InhibitoryNetwork(nn.Module):
    def __init__(self, num_layers, neuron_size, device='cpu'):
        super().__init__()
        self.layers = nn.ModuleList([
            InhibitoryNeuronLayer(neuron_size, device=device)
            for _ in range(num_layers)
        ])
        self.attn = InhibitorySelfAttention(neuron_size)
        self.device = device

    def forward(self, input_signals):
        x = input_signals
        inhibition_scores = []

        # Store each layer's inhibition for attention
        for layer in self.layers:
            x = layer(x + torch.randn_like(x) * 0.01)  # Add slight noise to break collapse
            inhibition_scores.append(torch.sigmoid(layer.inhibition.detach()))

        # Stack for attention
        inhibition_tensor = torch.stack(inhibition_scores).to(self.device)  # [num_layers, neuron_size]
        #print(f"inhibition_tensor shape before attention: {inhibition_tensor.shape}")
        # Mean inhibition per layer
        inhibition_level = inhibition_tensor.mean(dim=1)  # [num_layers]
        attn_output = self.attn(inhibition_tensor, inhibition_level)  # send [L, N] and [L]
        x = x + torch.mean(attn_output, dim=0)  # final fused output

        return x

    def reinforce_all(self, activations, reward_signal):
        for layer in self.layers:
            layer.reinforce(activations, reward_signal)

# Simulation
def multi_layer_simulation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InhibitoryNetwork(num_layers=3, neuron_size=10, device=device).to(device)

    # Target activation profile
    target_neurons = torch.randint(0, 2, (10,), device=device)
    print(f"Target neuron profile: {target_neurons.cpu().numpy()}")

    for episode in range(10000):
        input_signals = torch.rand(10, device=device)
        output = model(input_signals)
        decisions = (output > 0.5).float()
        reward = -torch.sum(torch.abs(decisions - target_neurons)).item()
        model.reinforce_all(output, reward)

        if episode % 500 == 0 or episode == 9999:
            print(f"Episode {episode}: reward = {reward:.2f}")

    # Show final inhibition landscape
    for i, layer in enumerate(model.layers):
        print(f"Layer {i} inhibition: {layer.inhibition.detach().cpu().numpy()}")

if __name__ == "__main__":
    multi_layer_simulation()
