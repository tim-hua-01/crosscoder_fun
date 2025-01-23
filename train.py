# %%
from utils import *
from trainer import Trainer
# %%
device = 'cuda:0'

base_model = HookedTransformer.from_pretrained(
    "pythia-160m-deduped", 
    device=device, 
)

checkpoint_mid_model = HookedTransformer.from_pretrained(
    "EleutherAI/pythia-160m-deduped", 
    device=device, 
    checkpoint_value = 512
)


import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from pathlib import Path

def compile_all_tokens(sequence_length=256, batch_size=512, max_batches=1000, device='cuda:0', save_dir=None):
    """
    Iterates through the pile_dedup_sample dataset and compiles a new all_tokens tensor.
    If a saved tensor exists at save_dir, loads that instead of recomputing.

    Args:
        sequence_length (int): The length of each token sequence.
        batch_size (int): The number of sequences per batch.
        max_batches (int, optional): The maximum number of batches to process. Defaults to None.
        device (str): The device to store the tensor on.
        save_dir (str, optional): Directory to save/load tensor from. If None, uses default name.

    Returns:
        torch.Tensor: A tensor containing all tokenized sequences.
    """
    # Create default save path if none provided
    if save_dir is None:
        save_dir = f"all_tokens_seq{sequence_length}_batch{batch_size}_max{max_batches}.pt"
    save_path = Path(save_dir)

    # Check if tensor already exists
    if save_path.exists():
        print(f"Loading existing tensor from {save_path}")
        return torch.load(save_path).to(device)

    # Load dataset
    pile_dedup_sample = load_dataset('EleutherAI/the_pile_deduplicated', streaming=True, split='train')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m-deduped')

    all_token_batches = []

    current_batch = []
    batch_count = 0

    print("Compiling all_tokens tensor...")
    pbar = tqdm(total=max_batches)
    for sample in pile_dedup_sample:
        # Tokenize the text
        tokens = tokenizer.encode(sample['text'])
        
        # Get as many sequence_length chunks as possible from tokens
        for i in range(0, len(tokens) - sequence_length + 1, sequence_length):
            current_batch.append(tokens[i:i + sequence_length])
            
            # If batch is full, add to all_token_batches
            if len(current_batch) == batch_size:
                batch_tensor = torch.tensor(current_batch, dtype=torch.int32)
                all_token_batches.append(batch_tensor)
                current_batch = []
                batch_count += 1
                pbar.update(1)
                # Check if we've reached the maximum number of batches
                if max_batches and batch_count >= max_batches:
                    break
        
        # Break outer loop if we've hit max_batches
        if max_batches and batch_count >= max_batches:
            break

    pbar.close()

    # Handle the last batch if it's not empty
    if current_batch:
        batch_tensor = torch.tensor(current_batch, dtype=torch.int32)
        all_token_batches.append(batch_tensor)

    # Concatenate all batches into a single tensor
    all_tokens = torch.cat(all_token_batches, dim=0).to(device)

    # Save tensor
    print(f"Saving tensor to {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_tokens.cpu(), save_path)

    print(f"Compiled all_tokens tensor with shape: {all_tokens.shape}")
    return all_tokens


# %%
tokens_no_bos = compile_all_tokens()
tokens_with_bos = torch.cat([torch.zeros(tokens_no_bos.shape[0], 1, device=tokens_no_bos.device, dtype=torch.int32), tokens_no_bos[:, :-1]], dim=1)

# %%
default_cfg = {
    "seed": 49,
    "batch_size": 1024,
    "buffer_mult": 512,
    "lr": 5e-5,
    "num_tokens": 3_000_000,#should probs be 32_000_000
    "l1_coeff": 1,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    "dict_size": 768*8*2,
    "seq_len": 256,
    "enc_dtype": "fp32",
    "model_name": "pythia-160m-deduped",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 16,
    "log_every": 100,
    "save_every": 30000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.5.hook_resid_pre",
    "wandb_project": "crosscoder-fun",
    "wandb_run_name": "no_more_halfass_buffering_with_asserts",
}
cfg = arg_parse_update_cfg(default_cfg)

if __name__ == "__main__":
    trainer = Trainer(cfg, base_model, checkpoint_mid_model, tokens_with_bos)
    print("Training...")
    trainer.train()
# %%