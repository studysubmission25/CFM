import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from pykeen.triples import TriplesFactory

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Positive Dataset
df_positive = pd.read_csv('positive_dataset.csv')
df_positive = df_positive[['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI']].sort_values(
    by=['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI']).reset_index(drop=True)

triples_factory_positive = TriplesFactory.from_labeled_triples(triples=df_positive.values)
entity_to_id_positive = triples_factory_positive.entity_to_id

# Load Original Embeddings
original_numpy = np.load("original_node_embeddings.npy")
original_embeddings = torch.tensor(original_numpy).to(device)
original_embeddings = nn.functional.normalize(original_embeddings, dim=1)

# Define Flow Matching Model
class Block(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, channels_data=100, layers=5, channels=512, channels_t=512):  
        super().__init__()
        self.channels_t = channels_t  
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(channels_t, channels)  
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(layers)])
        self.out_projection = nn.Linear(channels, channels_data)

    def gen_t_embedding(self, t, max_positions=10000):
        t = t * max_positions
        half_dim = self.channels_t // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        emb = emb.view(t.shape[0], -1)

        if emb.shape[1] < self.channels_t:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], self.channels_t - emb.shape[1]), device=t.device)], dim=1)

        return emb

    def forward(self, x, t):
        x = self.in_projection(x)
        t = self.gen_t_embedding(t)  
        t = self.t_projection(t)  
        x = x + t
        x = self.blocks(x)
        x = self.out_projection(x)
        return x

# Training Function
def train_flow_matching(model, optimizer, x_clean, epochs=100000, batch_size=512):
    model.train()
    dataloader = torch.utils.data.DataLoader(x_clean, batch_size=batch_size, shuffle=True)
    pbar = tqdm(range(epochs), position=0, leave=True)
    
    for i in pbar:
        x1 = next(iter(dataloader)).to(device)
        x0 = torch.randn_like(x1).to(device)
        t = torch.rand(x1.size(0), device=device)

        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        target = x1 - x0

        pred = model(xt, t)
        loss = ((target - pred) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

# Denoising Function
def denoise_embeddings(model, x_noisy, steps=50):
    model.eval()
    x_t = x_noisy.clone()
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([1 - i / steps]).float().to(x_noisy.device)
            t = t.repeat(x_noisy.shape[0], 1)
            flow = model(x_t, t)
            x_t = x_t + flow / steps
    return x_t

# Initialize and Train Flow Matching Model
input_dim = original_embeddings.shape[1]
denoiser = MLP(channels_data=input_dim, layers=5, channels=512, channels_t=512).to(device)
optimizer = optim.AdamW(denoiser.parameters(), lr=1e-4)

print("Training Flow Matching Model...")
train_flow_matching(denoiser, optimizer, original_embeddings, epochs=100000, batch_size=512)

# Denoise embeddings
print("Denoising embeddings...")
noise = torch.randn_like(original_embeddings).to(device)
#t = torch.rand((original_embeddings.shape[0], 1), device='cuda')  # Random t ∈ [0, 1] for each sample
t = 0.5
noised_embeddings = (1 - t) * noise + t * original_embeddings  # Example with t=0.5
torch.save(noised_embeddings.cpu(), "noised_embeddings.pt")
noised_embeddings_numpy = noised_embeddings.cpu().detach().numpy()
np.save('noised_node_embeddings.npy', noised_embeddings_numpy)

denoised_embeddings = denoise_embeddings(denoiser, noised_embeddings)

# Save the denoised embeddings
torch.save(denoised_embeddings.cpu(), "denoised_embeddings.pt")

# Save as NumPy array
denoised_embeddings_numpy = denoised_embeddings.cpu().detach().numpy()
np.save('denoised_node_embeddings.npy', denoised_embeddings_numpy)

print("Denoising complete! Embeddings saved.")

