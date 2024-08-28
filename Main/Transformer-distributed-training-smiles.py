import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import pandas as pd
import os

# Initialize distributed training
def init_distributed_training():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

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
smiles_list = load_chembl_data('chembl_smiles.csv')
smiles_list = [augment_smiles(smi) for smi in smiles_list]
tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
dataset = SMILESDataset(smiles_list, tokenizer)
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler)

# Initialize model
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

# Training loop with gradient accumulation
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()
accumulation_steps = 4  # Number of steps to accumulate gradients
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

# Save the model
if dist.get_rank() == 0:  # Save only on the main process
    model.module.save_pretrained('./chembl_transformer')
    tokenizer.save_pretrained('./chembl_transformer')

# Clean up
dist.destroy_process_group()

if __name__ == '__main__':
    init_distributed_training()

