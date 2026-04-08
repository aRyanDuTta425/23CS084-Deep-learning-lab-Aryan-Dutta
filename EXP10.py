import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ===== Load Dataset =====
with open("/Users/aryandutta/aryan jee/DL LAB ARYAN DUTTA 23CS084/text.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()[:100000]   # more data = better

chars = sorted(list(set(text)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# ===== Hyperparams =====
vocab_size = len(chars)
embed_size = 256
num_heads = 8
num_layers = 4
seq_len = 64
batch_size = 64

# ===== Batch =====
def get_batch():
    idx = torch.randint(0, len(data)-seq_len-1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in idx])
    y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y

# ===== Model =====
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, embed_size))

        decoder_layer = nn.TransformerDecoderLayer(
            embed_size, num_heads, batch_first=True, dropout=0.2
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos[:, :T, :]

        # causal mask (VERY IMPORTANT)
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)

        x = self.decoder(x, x, tgt_mask=mask)
        return self.fc(x)

model = GPT()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# ===== Text Generation =====
def generate(model, start="the ", length=100):
    model.eval()
    input_seq = torch.tensor([stoi[c] for c in start]).unsqueeze(0)

    for _ in range(length):
        x = input_seq[:, -seq_len:]
        out = model(x)

        probs = F.softmax(out[0, -1], dim=0)
        next_char = torch.multinomial(probs, 1).item()  # better than argmax

        input_seq = torch.cat([input_seq, torch.tensor([[next_char]])], dim=1)

    return "".join([itos[int(i)] for i in input_seq[0]])

# ===== Training =====
for epoch in range(10):
    model.train()
    total_loss = 0
    total_acc = 0

    for step in range(100):   # multiple batches per epoch
        x, y = get_batch()

        out = model(x)
        loss = criterion(out.reshape(-1, vocab_size), y.reshape(-1))

        preds = torch.argmax(out, dim=-1)
        acc = (preds == y).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    print(f"\nEpoch {epoch+1}")
    print(f"Avg Loss: {total_loss/100:.4f}")
    print(f"Avg Accuracy: {(total_acc/100)*100:.2f}%")

    print("Sample Text:")
    print(generate(model, "the ", 80))


#Results:
# Epoch 1
# Avg Loss: 2.4117
# Avg Accuracy: 30.90%
# Sample Text:
# the t ttetttt ttttttt ttttttttttttttttttttttttttttttttttttttttt tttttttttttttttttttt

# Epoch 2
# Avg Loss: 2.0625
# Avg Accuracy: 37.58%
# Sample Text:
# the theeedeeeteteeeedeeeeeie eeeeeeteteeeeeeeeeeteeeeeeeeeeeeeeeeeeeeeeeeeeeeee?emee

# Epoch 3
# Avg Loss: 1.9308
# Avg Accuracy: 40.45%
# Sample Text:
# the theee, hei th beiee eeeeeeeeeeeeeeeeeeeeeee eeeeeeeeeeeeee eeeeee eeeeeeeeeeeeee

# Epoch 4
# Avg Loss: 1.8291
# Avg Accuracy: 43.00%
# Sample Text:
# the teeeeeee-heeeeeeeeeeeeeeeeeeeee eeeteeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

# Epoch 5
# Avg Loss: 1.6996
# Avg Accuracy: 46.65%
# Sample Text:
# the thhhhhhhhhhhhhhhhhhhhhhhhohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh

# Epoch 6
# Avg Loss: 1.4829
# Avg Accuracy: 52.84%
# Sample Text:
# the hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh

# Epoch 7
# Avg Loss: 1.2974
# Avg Accuracy: 58.23%
# Sample Text:
# the that heh theee theeee tht thee that theee teetetth theeen theeeeeeett tttt theet

# Epoch 8
# Avg Loss: 1.1334
# Avg Accuracy: 63.09%
# Sample Text:
# the  the thhe thee thetet thee ththe ththett th theeth tthetthe thtththot the t thet

# Epoch 9
# Avg Loss: 0.9473
# Avg Accuracy: 68.84%
# Sample Text:
# the thethtttt tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt

# Epoch 10
# Avg Loss: 0.7826
# Avg Accuracy: 73.96%
# Sample Text:
# the thee th thhzhe thhe hhhhe thh thh thhhhhhh thhhhhhhh hhhhhhhhhh hhhhhhhhhhhhhhht
# (.venv) (base) ARYANs-MacBook-Air-2:DL LAB ARYAN DUTTA 23CS084 aryandutta$ 