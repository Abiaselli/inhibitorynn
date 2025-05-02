import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import random
from torchvision import transforms
from PIL import Image
import torchaudio
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
QUALIA_DIM = 256
VISUAL_INPUT_DIM = 768  # 3*16*16
AUDIO_FEATURES = 64
TEXT_INPUT_DIM = 128
HIDDEN_DIM = 512
MEMORY_DIM = 256
SEQ_LEN = 256
BATCH_SIZE = 64

# Qualia Preprocessing
class QualiaPreprocessor(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.visual_proj = nn.Linear(VISUAL_INPUT_DIM, QUALIA_DIM)
        self.audio_proj = nn.Linear(AUDIO_FEATURES, QUALIA_DIM)
        self.text_proj = nn.Linear(TEXT_INPUT_DIM, QUALIA_DIM)

    def text_to_onehot(self, text, vocab, max_len=TEXT_INPUT_DIM):
        vec = torch.zeros(max_len, dtype=torch.float32, device=self.device)
        for i, c in enumerate(text[:max_len]):
            idx = vocab.get(c, vocab.get('<unk>', 0))
            vec[idx] = 1.0
        return vec

    def waveform_to_features(self, waveform, sample_rate=16000, num_feats=AUDIO_FEATURES):
        mel = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=num_feats,
            melkwargs={'n_fft': 400, 'hop_length': 160}
        )(waveform.unsqueeze(0))
        return mel.mean(dim=-1).squeeze()

    def image_to_patch(self, img_pil):
        preprocess = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ])
        tensor_img = preprocess(img_pil).view(-1).to(self.device)
        return self.visual_proj(tensor_img)

    def forward(self, text, audio, image, vocab):
        tvec = self.text_proj(self.text_to_onehot(text, vocab))
        avec = self.audio_proj(self.waveform_to_features(audio))
        vvec = self.image_to_patch(image)
        return torch.stack([tvec, avec, vvec])

    def decode_qualia(self, qualia_tensor, domain='text', vocab_matrix=None):
        if domain == 'text' and vocab_matrix is not None:
            logits = vocab_matrix @ qualia_tensor.T
            return logits.T
        return qualia_tensor

# Dataset loader
def load_json_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_vocab(examples):
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for ex in examples:
        for c in ex['input'] + ex['output']:
            if c not in vocab:
                vocab[c] = idx
                idx += 1
    return vocab

def encode_text(text, vocab, max_len=SEQ_LEN):
    vec = torch.zeros((max_len, len(vocab)))
    for i, c in enumerate(text[:max_len]):
        idx = vocab.get(c, vocab['<unk>'])
        vec[i, idx] = 1.0
    return vec

def pad_or_trim(tensor, target_len):
    if tensor.shape[0] > target_len:
        return tensor[:target_len]
    elif tensor.shape[0] < target_len:
        pad = torch.zeros((target_len - tensor.shape[0], tensor.shape[1]), device=tensor.device)
        return torch.cat([tensor, pad], dim=0)
    return tensor

# Transplanting Core Weights into Larger Models
def transplant_weights(small_model_path, large_model):
    small_state = torch.load(small_model_path)
    new_state = large_model.state_dict()
    for k in new_state:
        if k in small_state and small_state[k].shape == new_state[k].shape:
            new_state[k] = small_state[k]
    large_model.load_state_dict(new_state)
    print("Weights transplanted.")

def auto_transplant_if_needed(model, checkpoint_path="model_weights.pth", strict_modules=("fc1", "fc2", "inhibitor", "encoder", "memory_attn")):
    if not os.path.exists(checkpoint_path):
        print(f"No saved model at {checkpoint_path}. Starting fresh.")
        return

    try:
        saved_state = torch.load(checkpoint_path)
        current_state = model.get_core_weights()
        transplantable = {}

        for key in current_state:
            # Check if this key belongs to a strict module
            if not any(key.startswith(mod) for mod in strict_modules):
                continue

            if key in saved_state:
                saved_tensor = saved_state[key]
                current_tensor = current_state[key]

                if saved_tensor.shape == current_tensor.shape:
                    transplantable[key] = saved_tensor
                    print(f"[=] Loaded '{key}' unchanged.")
                elif all(s <= c for s, c in zip(saved_tensor.shape, current_tensor.shape)):
                    slices = tuple(slice(0, s) for s in saved_tensor.shape)
                    padded = current_tensor.clone()
                    padded[slices] = saved_tensor
                    transplantable[key] = padded
                    print(f"[~] Padded '{key}' from {saved_tensor.shape} → {current_tensor.shape}")
                else:
                    print(f"[x] Skipped '{key}' (saved: {saved_tensor.shape}, current: {current_tensor.shape})")
            else:
                print(f"[x] Key '{key}' not found in saved weights.")

        model.load_core_weights(transplantable)
        print("Transplant complete.")
    except Exception as e:
        print(f"[!!] Transplant failed: {e}")


# Save model to file
def save_model(model, path="model_weights.pth"):
    torch.save(model.get_core_weights(), path)
    print(f"Model weights saved to {path}")

# Load model from file
def load_model(model, path="model_weights.pth"):
    try:
        state = torch.load(path)
        model.load_core_weights(state)
        print(f"Model weights loaded from {path}")
    except Exception as e:
        print(f"Failed to load model: {e}")

# Test Inference
def test_inference(model, vocab, prompt, max_len, device):
    model.eval()
    idx_to_char = {v: k for k, v in vocab.items()}
    with torch.no_grad():
        # Encode input prompt to one-hot [1, SEQ_LEN, vocab_size]
        vec = pad_or_trim(encode_text(prompt, vocab, max_len), max_len).unsqueeze(0).to(device)
        
        # Predict output
        out = model(vec)  # [1, SEQ_LEN, vocab_size]
        pred_ids = out.argmax(dim=-1).squeeze(0).tolist()

        # Decode prediction to text
        decoded = ''.join([idx_to_char.get(i, '?') for i in pred_ids if i in idx_to_char])
        print(f"\n[TEST OUTPUT]\nPrompt: {prompt}\nResponse: {decoded.strip()}\n")

# Model
class InhibitorySeq2SeqModel(nn.Module):
    def __init__(self, input_dim, output_dim, qualia_dim=QUALIA_DIM, hidden_dim=HIDDEN_DIM, memory_dim=MEMORY_DIM):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, qualia_dim)  # matches vocab size
        self.encoder = nn.Linear(qualia_dim + memory_dim, hidden_dim)
        self.memory_attn = nn.Linear(hidden_dim, hidden_dim)
        self.inhibitor = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.memory = None  # initialized later

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x = self.input_proj(x)

        # Memory
        if self.memory is None or self.memory.shape[0] != batch_size:
            self.memory = torch.zeros((batch_size, MEMORY_DIM), device=x.device)

        mem = 0.9 * self.memory + 0.1 * x.mean(dim=1).detach()
        self.memory = mem

        mem_expanded = mem.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([x, mem_expanded], dim=-1).reshape(-1, QUALIA_DIM + MEMORY_DIM)

        encoded = self.encoder(combined)
        inhibited = self.inhibitor(encoded)
        out = self.decoder(inhibited)

        return out.view(batch_size, seq_len, -1)

    def get_core_weights(self):
        return {
            'encoder': self.encoder.state_dict(),
            'inhibitor': self.inhibitor.state_dict(),
            'decoder': self.decoder.state_dict(),
            'input_proj': self.input_proj.state_dict(),
            'memory_attn': self.memory_attn.state_dict()
        }

    def load_core_weights(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.inhibitor.load_state_dict(state_dict['inhibitor'])
        self.decoder.load_state_dict(state_dict['decoder'])
        self.input_proj.load_state_dict(state_dict['input_proj'])
        self.memory_attn.load_state_dict(state_dict['memory_attn'])

# Training loop
def train_seq2seq_batched(model, vocab, examples, batch_size, max_len, epochs, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    vocab_size = len(vocab)

    def collate_fn(batch):
        inputs = [pad_or_trim(encode_text(ex['input'], vocab, max_len), max_len) for ex in batch]
        outputs = [pad_or_trim(encode_text(ex['output'], vocab, max_len), max_len) for ex in batch]
        return torch.stack(inputs).to(device), torch.stack(outputs).to(device)

    dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        total_loss = 0.0
        for in_seq, out_seq in dataloader:
            in_seq = in_seq.transpose(0, 1)
            out_seq = out_seq.transpose(0, 1)
            pred_seq = model(in_seq)
            loss = loss_fn(pred_seq, out_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[Epoch {epoch}] Avg Batch Loss: {total_loss:.6f}")
            test_inference(model, vocab, "Find the slope of the line $3x+5y=20$.", max_len, device)

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] Loss: {total_loss:.6f}")
            test_inference(model, vocab, "Find the slope of the line $3x+5y=20$.", max_len, device)
            save_model(model, f"model_epoch{epoch}.pth")
            print(f"[Epoch {epoch}] Loss: {total_loss:.6f}")
            torch.save(model.state_dict(), "model_weights.pth")

# Main
def run_seq2seq_nlp_training():
    path = "C:/Users/abias/.cursor-tutor/vccdoe/mhlamodel/mhla/KANseriesNeuralNetwork-main/inhibitorynetwork/data/mathtest.json"
    data = load_json_dataset(path)
    examples = []
    for item in data:
        convo = item['conversations']
        if len(convo) == 2:
            examples.append({'input': convo[0]['value'], 'output': convo[1]['value']})
    vocab = build_vocab(examples)
    print(len(vocab))
    model = InhibitorySeq2SeqModel(input_dim=len(vocab), output_dim=len(vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    auto_transplant_if_needed(model, "model_weights.pth")
    model=model.to(device)
    train_seq2seq_batched(model, vocab, examples, batch_size=BATCH_SIZE, max_len=SEQ_LEN, epochs=1000, device='cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    run_seq2seq_nlp_training()