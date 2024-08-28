################################
#Running the Script
#torchrun --nproc_per_node=<number_of_gpus> <script_name>.py\
#--metric cosine --queries "CCO" "O=C(C)Oc1ccccc1C(=O)O"
################################

####################
#Amirhossein Taghavi
#UF Scripps
#####################



import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import pandas as pd
from rdkit import Chem
import faiss
import numpy as np
import argparse

# Initialize distributed training
def init_distributed_training():
    # Ensure the environment variable for world size and rank are set
    if 'LOCAL_RANK' not in os.environ or 'RANK' not in os.environ:
        raise ValueError("Environment variables 'LOCAL_RANK' and 'RANK' must be set for distributed training")
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    # Print debug information
    print("Distributed Training Initialized")
    print("LOCAL_RANK:", os.environ['LOCAL_RANK'])
    print("RANK:", os.environ['RANK'])
    print("WORLD_SIZE:", os.environ.get('WORLD_SIZE'))

# Load ChEMBL dataset
def load_chembl_data(filepath):
    df = pd.read_csv(filepath)
    return df['smiles'].tolist()

# Custom Dataset Class
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=128):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        tokens = self.tokenizer(smiles, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

# Data augmentation (random SMILES)
def augment_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, doRandom=True)

# Load and tokenize data
def get_smiles_list_and_tokenizer(filepath):
    smiles_list = load_chembl_data(filepath)
    smiles_list = [augment_smiles(smi) for smi in smiles_list]
    tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    return smiles_list, tokenizer

# Training function
def train_model(smiles_list, tokenizer):
    dataset = SMILESDataset(smiles_list, tokenizer)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler)

    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=128
    )
    model = RobertaForMaskedLM(config)
    model.cuda()
    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    accumulation_steps = 4
    global_step = 0

    for epoch in range(3):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            input_ids, attention_masks = batch
            input_ids = input_ids.cuda()
            attention_masks = attention_masks.cuda()

            outputs = model(input_ids, attention_mask=attention_masks, labels=input_ids)
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {step + 1}, Global Step {global_step}, Loss: {loss.item()}")

    if dist.get_rank() == 0:
        model.module.save_pretrained('./chembl_transformer')
        tokenizer.save_pretrained('./chembl_transformer')

    dist.destroy_process_group()

# Extract embeddings from SMILES
def get_embeddings(smiles_list, tokenizer, model):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for smiles in smiles_list:
            tokens = tokenizer(smiles, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
            input_ids = tokens['input_ids'].cuda()
            attention_mask = tokens['attention_mask'].cuda()

            outputs = model(input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.cpu().numpy())
    
    return np.vstack(embeddings)

# Build FAISS index
def build_index(embeddings, metric):
    d = embeddings.shape[1]
    if metric == 'l2':
        index = faiss.IndexFlatL2(d)
    elif metric == 'cosine':
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
    elif metric == 'manhattan':
        index = faiss.IndexFlatL1(d)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    index.add(embeddings)
    return index

# Perform similarity search
def search_similar(smiles_queries, tokenizer, model, index, k=5):
    all_indices = []
    all_distances = []
    for smiles_query in smiles_queries:
        query_embedding = get_embeddings([smiles_query], tokenizer, model)
        if isinstance(index, faiss.IndexFlatIP):
            faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, k)
        all_indices.append(indices)
        all_distances.append(distances)
    return all_indices, all_distances

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model and perform similarity search on SMILES data.')
    parser.add_argument('--metric', choices=['l2', 'cosine', 'manhattan'], default='l2', help='Distance metric for similarity search')
    parser.add_argument('--queries', type=str, nargs='+', help='List of SMILES strings for querying')
    args = parser.parse_args()

    # Initialize distributed training
    init_distributed_training()

    # Training phase
    smiles_list, tokenizer = get_smiles_list_and_tokenizer('chembl_smiles.csv')
    train_model(smiles_list, tokenizer)

    # Load the trained model and tokenizer
    model = RobertaForMaskedLM.from_pretrained('./chembl_transformer')
    tokenizer = RobertaTokenizer.from_pretrained('./chembl_transformer')
    model.cuda()
    model.eval()

    # Extract embeddings and build index
    embeddings = get_embeddings(smiles_list, tokenizer, model)
    index = build_index(embeddings, args.metric)

    # Perform similarity search
    query_smiles_list = args.queries
    indices, distances = search_similar(query_smiles_list, tokenizer, model, index)
    for i, query in enumerate(query_smiles_list):
        print(f"Query SMILES: {query}")
        print("Indices of similar molecules:", indices[i])
        print("Distances to similar molecules:", distances[i])

