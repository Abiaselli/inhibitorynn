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

class NLPInhibitionSequence(nn.Module):
    def __init__(self, token_dim, hidden_dim, output_dim, seq_len=3):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Linear(token_dim, hidden_dim)
        self.inhibitor = InhibitionNeuron(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, qualia_sequence):  # [seq_len, token_dim]
        outputs = []
        for token in qualia_sequence:
            x = self.encoder(token)
            x = self.inhibitor(x)
            out = self.decoder(x)
            outputs.append(out)
        return torch.stack(outputs)  # shape: [seq_len, output_dim]

if __name__ == "__main__":
    tokenizer = QualiaTokenizer()
    qualia_model = InhibitionNet()
    byte_model = ByteMappingNet()

    optimizer_q = optim.Adam(qualia_model.parameters(), lr=1e-3)
    optimizer_b = optim.Adam(byte_model.parameters(), lr=1e-3)

    loss_fn = nn.BCEWithLogitsLoss()  # or nn.MSELoss()

    # Inputs
    word = "DOG"
    qualia_seq = torch.stack([tokenizer.encode(c, "text") for c in word])

    # Targets
    byte_targets = [ord(c) for c in word]
    byte_seq = torch.stack([
        torch.tensor([(b >> i) & 1 for i in range(8)], dtype=torch.float32)
        for b in byte_targets
    ])

    # Training
    model = NLPInhibitionSequence(token_dim=32, hidden_dim=64, output_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(qualia_seq)
        loss = loss_fn(output, byte_seq)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
