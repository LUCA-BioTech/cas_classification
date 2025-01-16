#!/usr/bin/env python
import argparse
import glob
import os
import pickle

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import numpy as np
import pandas as pd
import torch
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers import (
    EsmForMaskedLM,
    EsmForSequenceClassification,
    EsmModel,
    EsmTokenizer,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# from transformers.deepspeed import HfDeepSpeedConfig

import deepspeed
from pyfaidx import Fasta

def main(args):
    fasta_dir = args.fasta_dir
    model_name = args.model_name
    output = args.output
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    deepspeed.init_distributed("nccl")
    rank = dist.get_rank()

    tokenizer = EsmTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model_hidden_size = config.hidden_size
    train_batch_size = 1 * world_size

    ds_config = {
        "fp16": {
            "enabled": True
        },
        "bf16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": 3,
        },
        "train_micro_batch_size_per_gpu": 8,
        "compression_training":{
            "no_compress": True
            }
    }
    #ds_config["zero_optimization"]["offload_param"] = dict(device="cpu", pin_memory=True)
    ds_config["zero_optimization"]["offload_param"] = dict(device="cpu")
    # dschf = HfDeepSpeedConfig(ds_config)  # this tells from_pretrained to instantiate directly on gpus
    model = EsmModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.eval()
    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module

    seqs = extract_seqs(fasta_dir)

    embeddings = generate_embeddings(tokenizer, model, seqs['seq'], max_seq_len, batch_size)
    seqs['embeddings'] = embeddings.cpu().numpy().tolist()
    data = pd.DataFrame(seqs)
    pickle.dump(data, open(output, 'wb'))

def generate_embeddings(tokenizer, model, seqs, max_seq_len, batch_size):
    if max_seq_len is None:
        inputs = tokenizer.batch_encode_plus([str(seq) for seq in seqs],
                add_special_tokens=True, padding="longest", truncation=True, return_tensors="pt")
    else:
        inputs = tokenizer.batch_encode_plus([str(seq) for seq in seqs],
                add_special_tokens=True, padding="max_length", max_length=max_seq_len, truncation=True, return_tensors="pt")
    for t in inputs:
        if torch.is_tensor(inputs[t]):
            inputs[t] = inputs[t].to(torch.cuda.current_device())

    all_embeddings = None
    with torch.no_grad():
        # iterate over inputs with batch size
        for i in tqdm(range(0, len(seqs), batch_size)):
            batch = {k: v[i:i+batch_size] for k, v in inputs.items()}
            outputs = model(**batch)
            embeddings = outputs[0][:, 0, :]
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
        # outputs = model(**inputs)
        # embeddings = outputs[0][:, 0, :].detach().cpu().numpy()
        # if all_embeddings is None:
        #     all_embeddings = embeddings
        # else:
        #     all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    return all_embeddings

def extract_seqs(fasta_dir):
    fasta_files = glob.glob(fasta_dir + '/*.fasta') + glob.glob(fasta_dir + '/*.faa')
    fasta_files = fasta_files
    seqs = {'name': [], 'seq': [], 'category': []}
    for fasta_file in fasta_files:
        category = fasta_file.split('/')[-1].split('.')[0]
        for seq in Fasta(fasta_file):
            seqs['name'].append(seq.name)
            seqs['seq'].append(str(seq))
            seqs['category'].append(category)
    return seqs

def create_parser():
    parser = argparse.ArgumentParser(description='Build embeddings')
    parser.add_argument('-d', '--fasta_dir', required=True, help='Directory containing fasta files')
    parser.add_argument('-m', '--model_name', help='model', default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('-l', '--max_seq_len', type=int, help='max sequence length')
    parser.add_argument('--batch_size', type=int, help='batch size', default=10)
    parser.add_argument('-o', '--output', required=True, help='output embedding file')
    parser.add_argument('--local_rank', type=int)

    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
