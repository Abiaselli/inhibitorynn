import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random

# -- Tokenizer --
class NLPQualiaTokenizer:
    def __init__(self):
        self.alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;!?-+/=()[]{}\\$%&^@#*:\"' \n"
        self.vocab = {ch: i for i, ch in enumerate(self.alphabet)}
        self.domain_tag = torch.tensor([1.0, 0.0])  # for NLP

    def encode_char(self, ch):
        one_hot = torch.zeros(len(self.vocab))
        if ch in self.vocab:
            one_hot[self.vocab[ch]] = 1.0
        return torch.cat([one_hot, self.domain_tag], dim=0)

    def encode_string(self, text, max_len):
        vecs = [self.encode_char(ch) for ch in text[:max_len]]
        while len(vecs) < max_len:
            vecs.append(torch.zeros_like(vecs[0]))  # pad to fixed length
        return torch.stack(vecs)

    def decode_tensor(self, vecs):
        idx_to_char = {v: k for k, v in self.vocab.items()}
        out = ""
        for vec in torch.sigmoid(vecs):
            ch_idx = vec[:-2].argmax().item()
            out += idx_to_char.get(ch_idx, '?')
        return out.strip()

# -- Inhibition Model --
class InhibitionNeuron(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.inhibition = nn.Parameter(torch.rand(size))
        self.threshold = nn.Parameter(torch.rand(size))

    def forward(self, x):
        collapse = torch.sigmoid(self.threshold - self.inhibition)
        return x * collapse

class InhibitorySeq2SeqModel(nn.Module):
    def __init__(self, qualia_dim, memory_dim=256, hidden_dim=512):
        super().__init__()
        self.memory_dim = memory_dim
        self.encoder = nn.Linear(qualia_dim + memory_dim, hidden_dim)
        self.inhibitor = InhibitionNeuron(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, qualia_dim)

    def forward(self, input_seq):
        memory = torch.zeros(self.memory_dim, device=input_seq.device)
        outputs = []
        for x in input_seq:
            xmem = torch.cat([x, memory])
            h = self.inhibitor(self.encoder(xmem))
            out = self.decoder(h)
            memory = h[:self.memory_dim].detach()
            outputs.append(out)
        return torch.stack(outputs)

# -- Dataset Loader --
def load_math_json(path, max_samples=100):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    for entry in data[:max_samples]:
        conv = entry["conversations"]
        q = next((c["value"] for c in conv if c["from"] == "human"), None)
        a = next((c["value"] for c in conv if c["from"] == "gpt"), None)
        if q and a:
            examples.append((q.strip(), a.strip()))
    return examples

# -- Training Loop with Batching --
def train_seq2seq_batched(model, tokenizer, examples, batch_size=4, max_len=128, epochs=1000, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.to(device)

    for epoch in range(epochs + 1):
        total_loss = 0
        random.shuffle(examples)

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            batch_loss = 0

            for q, a in batch:
                in_seq = tokenizer.encode_string(q, max_len).to(device)
                out_seq = tokenizer.encode_string(a, max_len).to(device)

                pred_seq = model(in_seq)
                min_len = min(pred_seq.size(0), out_seq.size(0))
                loss = loss_fn(pred_seq[:min_len], out_seq[:min_len])
                batch_loss += loss

            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"[Epoch {epoch}] Avg Batch Loss: {total_loss:.6f}")
            test_inference(model, tokenizer, "Find the slope of the line $3x+5y=20$.", max_len, device)

# -- Test Inference --
def test_inference(model, tokenizer, prompt, max_len, device):
    model.eval()
    with torch.no_grad():
        vec = tokenizer.encode_string(prompt, max_len).to(device)
        out = model(vec)
        decoded = tokenizer.decode_tensor(out)
        print(f"\n[TEST OUTPUT]\nPrompt: {prompt}\nResponse: {decoded}\n")

# -- Main Entry Point --
def run_seq2seq_nlp_training():
    path = "C:\Users\abias\alldatareasoning\iimath\1\infinityinstructmath (1).json"
    examples = load_math_json(path, max_samples=500)
    tokenizer = NLPQualiaTokenizer()
    qualia_dim = len(tokenizer.vocab) + 2
    model = InhibitorySeq2SeqModel(qualia_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_seq2seq_batched(model, tokenizer, examples, batch_size=20, max_len=1024, epochs=1000, device=device)

# -- Run the script --
run_seq2seq_nlp_training()

