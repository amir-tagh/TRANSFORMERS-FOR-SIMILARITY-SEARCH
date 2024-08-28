import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load pre-trained ChemBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')

# Function to tokenize SMILES
def tokenize_smiles(smiles):
    tokens = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
    return tokens

# Function to get embeddings
def get_embeddings(smiles_list):
    tokens = tokenize_smiles(smiles_list)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to calculate similarity
def calculate_similarity(smiles_list1, smiles_list2):
    embeddings1 = get_embeddings(smiles_list1)
    embeddings2 = get_embeddings(smiles_list2)
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    return similarity_matrix

# Example usage
if __name__ == "__main__":
    smiles1 = ["CCO", "CCN"]
    smiles2 = ["CCC", "CCO"]
    
    similarity = calculate_similarity(smiles1, smiles2)
    print("Similarity Matrix:\n", similarity)

