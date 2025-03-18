#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:18:45 2025

@author: betulerkantarci
"""


import pandas as pd
from pykeen.pipeline import pipeline
import torch
from pykeen.triples import TriplesFactory
import random
import numpy as np
from pykeen.models import RGCN

# Set random seed for reproducibility
seed = 1551321751  
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior in CuDNN
    torch.backends.cudnn.benchmark = False  # Disables automatic optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
# Load and sort dataset
df_noised = pd.read_csv('noised_dataset.csv')
df_noised = df_noised.sample(frac=1).reset_index(drop=True)
df_noised = df_noised[['SUBJECT_CUI', 'PREDICATE', 'OBJECT_CUI']]

# Prepare triples
triples_factory = TriplesFactory.from_labeled_triples(triples=df_noised.values)

# Ensure split is deterministic
training, testing, validation = triples_factory.split(ratios=(.8, .1, .1), random_state=seed)

# Run PyKEEN pipeline
result = pipeline(
    training=triples_factory,
    testing=testing,
    validation=validation,
    model='RGCN',
    model_kwargs={
        'embedding_dim': 100,
        'num_layers': 1,
    },
    training_kwargs={
        'num_epochs': 100,
    },
    stopper='early',
    device=device,
    random_seed=seed 
)

# Save the trained model
model = result.model
model_path = 'noised_rgcn_model.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Save embeddings
node_embeddings = model.entity_representations[0]
node_embeddings_numpy = node_embeddings().cpu().detach().numpy()

np.save('noised_node_embeddings.npy', node_embeddings_numpy)

print(f'Node Embeddings Shape: {node_embeddings_numpy.shape}')
print(f'Embedding Dimension: {node_embeddings().shape[1]}')

# Verify node embeddings
node_names = list(result.training.entity_to_id.keys())
for name in node_names[:5]:
    entity_id = result.training.entity_to_id[name]
    print(f'Embedding for node "{name}" (ID: {entity_id}): {node_embeddings_numpy[entity_id]}')

    
    
"""
# Loading the model

from pykeen.pipeline import pipeline
from pykeen.models import RGCN
import torch


model_path = 'noised_rgcn_model.pt'

loaded_model = RGCN(
    triples_factory=combined_factory,
    embedding_dim=100,
     num_layers=1,
   # interaction="DistMult",
)
loaded_model.load_state_dict(torch.load(model_path))
print("Model reloaded successfully")
"""
