import torch
import torch.nn as nn
import torch.optim as optim


MAX_INHIBITION = 2**63 - 1

class IntegerInhibitoryNeuronLayer(torch.nn.Module):
    def __init__(self, size, device='cpu'):
        super().__init__()
        self.device = device
        self.size = size
        self.inhibition = torch.full((size,), MAX_INHIBITION, dtype=torch.int64, device=device)

    def forward(self, input_signals):
        # input_signals assumed to be small integers
        self.inhibition = torch.clamp(self.inhibition - input_signals, 0, MAX_INHIBITION)
        # lower inhibition = more active
        activations = MAX_INHIBITION - self.inhibition
        return activations

class InhibitoryNeuronLayer(nn.Module):
    def __init__(self, size, device='cpu'):
        super().__init__()
        self.device = device
        self.size = size
        
        # Inhibition values (starts near zero = fully active)
        self.inhibition = nn.Parameter(torch.zeros(size, dtype=torch.float32, device=device))
        
        # Learning rates for reward and punishment
        self.reward_lr = 0.01
        self.punishment_lr = 0.02

    def forward(self, input_signals):
        # Apply inhibition
        activations = (1.0 - torch.sigmoid(self.inhibition)) * input_signals
        return activations

    def reinforce(self, activations, reward_signal):
        """
        Modify inhibition based on a reward signal:
        - Positive reward: reduce inhibition (make neuron easier to activate)
        - Negative reward: increase inhibition (make neuron harder to activate)
        """
        with torch.no_grad():
            if reward_signal > 0:
                # Reward: Decrease inhibition proportional to activation
                self.inhibition -= self.reward_lr * activations
            else:
                # Punishment: Increase inhibition proportional to activation
                self.inhibition += self.punishment_lr * activations

            # Clamp inhibition to reasonable range
            self.inhibition.clamp_(-10.0, 10.0)

class InhibitorySelfAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.value_proj = torch.nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, inhibition_scores):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # Standard dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        
        # Apply inhibitory influence: higher inhibition reduces attention
        attn_scores = attn_scores - inhibition_scores.unsqueeze(1)  # broadcast inhibition
        
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output


# Example Simulation
def toy_simulation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neuron_layer = InhibitoryNeuronLayer(size=10, device=device)
    
    # Create a dummy optimizer (not used for learning inhibition here, but could extend later)
    optimizer = optim.Adam(neuron_layer.parameters(), lr=0.00001)

    # Simulate environment: which neurons are "good" to activate
    target_neurons = torch.randint(0, 2, (10,), device=device)  # Randomly assign good (1) vs bad (0)

    print(f"Target neuron profile (good=1, bad=0): {target_neurons.cpu().numpy()}")
    
    for episode in range(50000):
        optimizer.zero_grad()

        # Random input stimulation
        input_signals = torch.rand(10, device=device)

        # Forward pass through inhibition
        activations = neuron_layer(input_signals)

        # Decision: which neurons "fire"
        decisions = (activations > 0.5).float()

        # Success is activating only correct neurons (matching target_neurons)
        reward_signal = -torch.sum(torch.abs(decisions - target_neurons)).item()

        # Provide reward or punishment
        neuron_layer.reinforce(activations, reward_signal)

        # Monitor learning
        if episode % 5 == 0 or episode == 49:
            print(f"Episode {episode}: reward={reward_signal:.2f}")

    print("\nFinal inhibition values:", neuron_layer.inhibition.detach().cpu().numpy())

if __name__ == "__main__":
    toy_simulation()
