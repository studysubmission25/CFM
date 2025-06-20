#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:53:46 2025
"""

import pandas as pd
from Bio import Entrez, Medline
import time

# Replace with your email (required by NCBI)
Entrez.email = ""
# Optional: Entrez.api_key = "your_ncbi_api_key"

# --- Load your CSV file ---
df = pd.read_csv("diffusion_model/model_comparison/osimertinib/osimertinib_pass_drugnames.csv") # ensure it has 'CUI' and 'Name'

# Prepare list to store results
article_counts = []

# Iterate over each drug
for name in df["Name"]:
    query = f"osimertinib AND {name}"

    try:
        # Search PubMed
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=100)
        search_results = Entrez.read(search_handle)
        pmid_list = search_results["IdList"]
        count = len(pmid_list)

        # Optional: delay to avoid overloading the server
        time.sleep(0.4)

    except Exception as e:
        print(f"Error with {name}: {e}")
        count = 0

    article_counts.append(count)

# Add result to original DataFrame
df["PubMed_Count_with_osimertinib"] = article_counts
df = df.sort_values(by="PubMed_Count_with_osimertinib", ascending=False)

# Save updated CSV
df.to_csv("diffusion_model/model_comparison/osimertinib/osimertinib_pass_drugnames.csv", index=False)


print("✅ Updated CSV saved as 'updated_drug_list_with_counts.csv'")
