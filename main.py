import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load CSV files
companies_df = pd.read_csv("ml_insurance_challenge.csv")
taxonomy_df = pd.read_csv("insurance_taxonomy - insurance_taxonomy.csv")

# Prepare input text
company_texts = companies_df['description'].fillna('') + ' ' + companies_df['business_tags'].fillna('').astype(str)
taxonomy_labels = taxonomy_df['label'].dropna().unique().tolist()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
company_embeddings = model.encode(company_texts.tolist(), convert_to_tensor=True, show_progress_bar=True)
taxonomy_embeddings = model.encode(taxonomy_labels, convert_to_tensor=True, show_progress_bar=True)

# Compute cosine similarity and assign labels
threshold = 0.32
filtered_labels = []
for i in range(len(companies_df)):
    scores = util.cos_sim(company_embeddings[i], taxonomy_embeddings)[0]
    valid_indices = (scores >= threshold).nonzero(as_tuple=True)[0]
    labels = [taxonomy_labels[idx] for idx in valid_indices]
    filtered_labels.append(labels)

# Add new column to original dataframe
companies_df['insurance_label'] = filtered_labels

# Overwrite the same file with new column
companies_df.to_csv("ml_insurance_challenge.csv", index=False)

print("Updated 'ml_insurance_challenge.csv' with column 'insurance_label'")