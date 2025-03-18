#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow Matching Denoising for Embeddings
Created on Feb 17, 2025
@author: betulerkantarci
"""

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

# Load Noised Dataset
df_noised = pd.read_csv('noised_dataset.csv')
df_noised = df_noised[['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI']].sort_values(
    by=['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI']).reset_index(drop=True)

triples_factory_noised = TriplesFactory.from_labeled_triples(triples=df_noised.values)
entity_to_id_noised = triples_factory_noised.entity_to_id

if entity_to_id_positive == entity_to_id_noised:
    print("Entity mappings MATCH between positive and noised datasets!")

# Load Original and Noised Embeddings
#original_embeddings = torch.load("/home/ubuntu/ECMLPKDD/diffusion_model/original_rgcn_model.pt", map_location=device)
#original_embeddings = original_embeddings['entity_representations.0.entity_embeddings._embeddings.weight'].to(device)

#noised_embeddings = torch.load("/home/ubuntu/ECMLPKDD/diffusion_model/noised_rgcn_model.pt", map_location=device)
#noised_embeddings = noised_embeddings['entity_representations.0.entity_embeddings._embeddings.weight'].to(device)

# Load denoised embeddings from .npy file
original_numpy = np.load("original_node_embeddings.npy")
# Convert to PyTorch tensor and move to device (CPU/GPU)
original_embeddings = torch.tensor(original_numpy).to(device)

# Load denoised embeddings from .npy file
noised_numpy = np.load("noised_node_embeddings.npy")
# Convert to PyTorch tensor and move to device (CPU/GPU)
noised_embeddings = torch.tensor(noised_numpy).to(device)


# Normalize embeddings
original_embeddings = nn.functional.normalize(original_embeddings, dim=1)
noised_embeddings = nn.functional.normalize(noised_embeddings, dim=1)


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
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # (batch_size, 512)

        # 🔹 Ensure emb is always 2D (batch_size, channels_t)
        emb = emb.view(t.shape[0], -1)  # Reshape to ensure 2D tensor

        # 🔹 Ensure emb.shape[1] == channels_t
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
def train_flow_matching(model, optimizer, x_noisy, x_clean, epochs=100000, batch_size=512):
    model.train()
    dataset = torch.utils.data.TensorDataset(x_noisy, x_clean)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    pbar = tqdm(range(epochs), position=0, leave=True)
    losses = []

    for i in pbar:
        # Sample a batch
        x1 = x_clean[torch.randint(x_clean.size(0), (batch_size,))].to(device)
        x0 = x_noisy[torch.randint(x_noisy.size(0), (batch_size,))].to(device)  
        target = x1 - x0
        t = torch.rand(x1.size(0), device=device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        pred = model(xt, t)  

        loss = ((target - pred) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item()) 

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
train_flow_matching(denoiser, optimizer, noised_embeddings, original_embeddings, epochs=100, batch_size=512)

# Denoise embeddings
print("Denoising embeddings...")
denoised_embeddings = denoise_embeddings(denoiser, noised_embeddings)

# Save the denoised embeddings
torch.save(denoised_embeddings.cpu(), "denoised_embeddings.pt")

# Save as NumPy array
denoised_embeddings_numpy = denoised_embeddings.cpu().detach().numpy()
np.save('denoised_node_embeddings.npy', denoised_embeddings_numpy)

print("Denoising complete! Embeddings saved.")

