# Model building

goal: train a transformer model like ChemBERTa from scratch on the ChEMBL database or another dataset.

**1. Dataset Preparation**
   
Download ChEMBL: Obtain the ChEMBL dataset (e.g., via the ChEMBL web interface or FTP).

Preprocess: Extract and preprocess SMILES strings, ensuring data quality.

**2. Tokenization**

Use a SMILES-specific tokenizer or implement a custom tokenizer that handles the specific format of SMILES strings.

**3. Model Architecture**

Define a transformer-based model (like BERT or Roberta). For ChemBERTa, you'd typically use a RoBERTa architecture adapted for SMILES.

**5. Training the Model**

Training Loop: Implement a PyTorch training loop that feeds SMILES strings into the model, computes loss (e.g., cross-entropy for masked language modeling), and updates model weights using backpropagation.

# Key Features:

**Distributed Training**: The script uses PyTorch's Distributed Data Parallel (DDP) to enable distributed training across multiple GPUs.

**Gradient Accumulation**: The training loop accumulates gradients over multiple steps (accumulation_steps) before performing an optimizer step, which is useful when training large models with limited GPU memory.

**Data Augmentation**: A simple SMILES augmentation technique is added that generates random SMILES for each molecule to diversify the training data.

# Requirements:
NCCL backend: Used for distributed GPU communication.

Environment Variables: Set LOCAL_RANK in your distributed environment.

Distributed Training Setup: Make sure to run the script with torch.distributed.launch.

# Perform a similarity search:

Encode Molecules: Convert the SMILES strings of both the target and database molecules into embeddings using the trained model.

This involves passing the SMILES through the model to obtain their respective hidden states or pooled outputs.

Compute Similarity: Use a similarity metric, such as cosine similarity, to compare the embeddings of the target molecule with those of the database molecules.

Ranking: Rank the database molecules based on their similarity scores to the target molecule.

# Running the Script:

torchrun --nproc_per_node=<number_of_gpus> <script_name>.py --metric cosine --queries "CCO" "O=C(C)Oc1ccccc1C(=O)O"

## Acknowledgements
We thank the authors of following open-source packages:
- [torch](https://pytorch.org/)
- [faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- [RDkit](https://www.rdkit.org/)
- [numpy](https://numpy.org/)
  
