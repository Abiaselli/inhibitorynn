import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QualiaTokenizer:
    def __init__(self):
        self.structure_vocab = {c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
        self.phoneme_map = {c: [1, 0] if c in "AEIOU" else [0, 1] for c in self.structure_vocab}  # crude vowels/consonants
        self.visual_map = {c: [0.5, 0.5] for c in self.structure_vocab}  # grayscale placeholder
        self.domain_tags = {"text": [1, 0], "audio": [0, 1]}

    def encode(self, symbol: str, domain: str) -> torch.Tensor:
        structure = torch.nn.functional.one_hot(torch.tensor(self.structure_vocab[symbol]), num_classes=26).float()
        phoneme = torch.tensor(self.phoneme_map[symbol])
        visual = torch.tensor(self.visual_map[symbol])
        domain_tag = torch.tensor(self.domain_tags[domain])
        return torch.cat([structure, phoneme, visual, domain_tag], dim=0)

class InhibitionNeuron(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.inhibition = nn.Parameter(torch.rand(size))
        self.threshold = nn.Parameter(torch.rand(size))

    def forward(self, x):
        collapse_prob = torch.sigmoid(self.threshold - self.inhibition)
        return x * collapse_prob  # 1 = pass, 0 = collapse

class InhibitionNet(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, output_size=8):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.inhibitor = InhibitionNeuron(hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)  # 8-bit binary output

    def forward(self, x):
        x = self.encoder(x)
        x = self.inhibitor(x)
        return self.decoder(x)

class ByteMappingNet(nn.Module):
    def __init__(self, byte_size=8):
        super().__init__()
        self.encoder = nn.Linear(byte_size, 64)
        self.inhibitor = InhibitionNeuron(64)
        self.decoder = nn.Linear(64, byte_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.inhibitor(x)
        return self.decoder(x)

def train_step(model, input_tensor, target_tensor, optimizer, loss_fn):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = loss_fn(output, target_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()

class InhibitoryNeuronLayer(nn.Module):
    def __init__(self, size, device='cpu'):
        super().__init__()
        self.device = device
        self.inhibition = nn.Parameter(torch.randn(size, dtype=torch.float64, device=device))  # start with diversity
        self.reward_lr = 0.01
        self.punishment_lr = 0.1

    def forward(self, input_signals):
        activations = (1.0 - torch.sigmoid(self.inhibition)) * input_signals
        return activations

    def reinforce(self, activations, reward_signal):
        with torch.no_grad():
            if reward_signal > 0:
                self.inhibition -= self.reward_lr * activations
            else:
                self.inhibition += self.punishment_lr * activations
            self.inhibition.clamp_(0, 1.0)

class InhibitorySelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, inhibition_level):
        # x: [num_layers, neuron_size], inhibition_level: [num_layers]
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        #print(f"attn_scores shape: {attn_scores.shape}")
        #print(f"inhibition_scores shape: {inhibition_scores.shape}")
        #print(f"inhibition_scores unsqueezed shape: {inhibition_scores.unsqueeze(1).shape}")

        attn_scores = attn_scores - inhibition_level.unsqueeze(1)  # shape match: [num_layers, num_layers]
        
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
    model = InhibitoryNetwork(num_layers=10, neuron_size=100, device=device).to(device)
    model = model.double()
    # Target activation profile
    target_neurons = torch.randint(0, 2, (100,), device=device)
    print(f"Target neuron profile: {target_neurons.cpu().numpy()}")

    for episode in range(10000):
        input_signals = torch.rand(100, device=device)
        output = model(input_signals)
        decisions = (output > 0.5).double()
        reward = -torch.sum(torch.abs(decisions - target_neurons)).item()
        model.reinforce_all(output, reward)

        if episode % 500 == 0 or episode == 9999:
            print(f"Episode {episode}: reward = {reward:.2f}")

    # Show final inhibition landscape
    for i, layer in enumerate(model.layers):
        print(f"Layer {i} inhibition: {layer.inhibition.detach().cpu().numpy()}")

if __name__ == "__main__":
    tokenizer = QualiaTokenizer()
    qualia_model = InhibitionNet()
    byte_model = ByteMappingNet()

    optimizer_q = optim.Adam(qualia_model.parameters(), lr=1e-3)
    optimizer_b = optim.Adam(byte_model.parameters(), lr=1e-3)

    loss_fn = nn.BCEWithLogitsLoss()  # or nn.MSELoss()

    for epoch in range(1000):
        symbol = "A"
        domain = "text"
        qualia = tokenizer.encode(symbol, domain)
        byte = ord(symbol)
        byte_tensor = torch.tensor([(byte >> i) & 1 for i in range(8)], dtype=torch.float32)  # binary target

        # Qualia to byte
        loss_q = train_step(qualia_model, qualia, byte_tensor, optimizer_q, loss_fn)

        # Byte to byte
        input_byte = byte_tensor + torch.randn(8) * 0.05  # add noise for training
        loss_b = train_step(byte_model, input_byte, byte_tensor, optimizer_b, loss_fn)

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] Qualia→Byte loss: {loss_q:.4f} | Byte→Byte loss: {loss_b:.4f}")
