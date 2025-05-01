import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.io import wavfile
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---- Main model ----
class VisualNLPSentenceModel(nn.Module):
    def __init__(self, qualia_dim=12314, hidden_dim=128, out_dim=3):

        super().__init__()
        self.fc1 = nn.Linear(qualia_dim, hidden_dim)
        self.inhibitor = InhibitionNeuron(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_seq):
        return torch.stack([self.fc2(self.inhibitor(self.fc1(x))) for x in x_seq])

# ---- Simulated image (3 blocks: the, dog, runs) ----
def generate_fake_sentence_image(size=(3, 64, 64)):
    img = torch.zeros(size[0], size[1], size[2], 3)
    colors = [[1, 1, 1], [0.6, 0.4, 0.1], [0.1, 0.9, 0.1]]
    for i in range(3):
        img[i, :, :, :] = torch.tensor(colors[i])
    img = img.permute(1, 0, 2, 3).reshape(size[1], size[0] * size[2], 3)
    img = (img * 255).byte().numpy()
    return img

def save_fake_image():
    fake_img = generate_fake_sentence_image()
    img = Image.fromarray(fake_img)
    img = img.resize((192, 64))  # 3 blocks of 64×64 each
    img.save("the_dog_runs.png")


# ---- Extract visual blocks ----
def extract_visual_blocks(path, cols=3):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    block_w = w // cols
    return [
        torch.tensor(np.array(img.crop((i * block_w, 0, (i + 1) * block_w, h))).astype(np.float32)).view(-1) / 255.0
        for i in range(cols)
    ]

# ---- Encode qualia vector (nlp one-hot + image patch) ----
def encode_qualia(token, visual_patch):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    one_hot = torch.zeros(26)
    if token.lower() in alphabet:
        one_hot[alphabet.index(token.lower()[0])] = 1.0
    return torch.cat([one_hot, visual_patch])

def extract_blocks(image_path, rows=1, cols=3):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    block_w, block_h = w // cols, h // rows
    blocks = []

    for r in range(rows):
        for c in range(cols):
            box = (c*block_w, r*block_h, (c+1)*block_w, (r+1)*block_h)
            cropped = img.crop(box)
            blocks.append(torch.tensor(np.array(cropped)).float().view(-1) / 255.0)

    return blocks

def plot_inhibition_state(model):
    inhib = model.inhibitor.inhibition.detach().cpu().numpy()
    thresh = model.inhibitor.threshold.detach().cpu().numpy()
    collapse = torch.sigmoid(model.inhibitor.threshold - model.inhibitor.inhibition).detach().cpu().numpy()

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.title("Inhibition")
    plt.plot(inhib)
    plt.subplot(1, 3, 2)
    plt.title("Threshold")
    plt.plot(thresh)
    plt.subplot(1, 3, 3)
    plt.title("Collapse Probability")
    plt.plot(collapse)
    plt.tight_layout()
    plt.show()

# --- Inhibition Layer ---
class InhibitionNeuron(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.inhibition = nn.Parameter(torch.rand(size))
        self.threshold = nn.Parameter(torch.rand(size))

    def forward(self, x):
        collapse = torch.sigmoid(self.threshold - self.inhibition)
        return x * collapse

class InhibitionNeuronWithDecay(nn.Module):
    def __init__(self, size, decay=0.01):
        super().__init__()
        self.inhibition = nn.Parameter(torch.rand(size))
        self.threshold = nn.Parameter(torch.rand(size))
        self.decay_rate = decay
        self.baseline = nn.Parameter(torch.zeros(size), requires_grad=False)

    def forward(self, x):
        # Decay toward baseline
        with torch.no_grad():
            self.inhibition.data = self.inhibition.data * (1 - self.decay_rate) + self.baseline * self.decay_rate

        collapse = torch.sigmoid(self.threshold - self.inhibition)
        return x * collapse

# --- Multimodal Inhibition Network ---
class MultimodalInhibitoryConceptNet(nn.Module):
    def __init__(self, qualia_dim=256, hidden=128, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(qualia_dim, hidden)
        self.inhibitor = InhibitionNeuron(hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.inhibitor(x)
        return self.fc2(x)

# --- Synthetic Qualia Generator ---
def generate_qualia_vector(word, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # NLP (one-hot of each char, summed)
    struct = torch.zeros(26)
    for c in word:
        if c.isalpha():
            struct[ord(c.upper()) - ord('A')] += 1.0
    struct /= struct.sum()  # normalize

    # Audio (synthetic waveform: sin + noise)
    t = torch.linspace(0, 1, steps=64)
    waveform = torch.sin(2 * np.pi * (seed + 1) * t) + torch.randn(64) * 0.1

    # Visual (synthetic RGB patch flattened)
    rgb = torch.rand(4, 4, 3)
    flat_rgb = rgb.view(-1)

    return torch.cat([struct, waveform, flat_rgb])  # [26 + 64 + 48 = 138]

def get_image_bytes(img_path, num_pixels=16):
    image = Image.open(img_path).convert("RGB")
    image = T.Resize((4, 4))(image)
    flat = torch.tensor(image).view(-1, 3)[:num_pixels]  # RGB tuples
    return flat.float() / 255.0  # normalize to [0, 1]

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


# ---- Target labels ----
concepts = {
    "the": torch.tensor([1.0, 0.0, 0.0]),
    "dog": torch.tensor([0.0, 1.0, 0.0]),
    "runs": torch.tensor([0.0, 0.0, 1.0])
}
# ---- Main training loop ----
def train_sentence_model():
    save_fake_image()
    model = VisualNLPSentenceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    words = ["the", "dog", "runs"]
    img_blocks = extract_visual_blocks("the_dog_runs.png")
    qualia_seq = [encode_qualia(w, img_blocks[i]) for i, w in enumerate(words)]
    labels = [concepts[w] for w in words]

    for epoch in range(1001):
        optimizer.zero_grad()
        preds = model(qualia_seq)
        loss = sum(loss_fn(preds[i], labels[i]) for i in range(3))
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")
            for i, word in enumerate(words):
                probs = torch.sigmoid(preds[i]).detach().numpy()
                print(f"  {word:<4} → {np.round(probs, 3)}")

def get_audio_bytes(wav_path, max_len=64):
    sr, samples = wavfile.read(wav_path)  # typically int16
    samples = samples[:max_len]  # crop or pad
    normed = (samples / 32768.0).clip(-1, 1)  # normalize
    return torch.tensor(normed, dtype=torch.float32)  # [-1, 1] range

if __name__ == "__main__":

    train_sentence_model()
