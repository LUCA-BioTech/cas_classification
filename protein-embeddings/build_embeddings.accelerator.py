#!/usr/bin/env python
import os
import argparse
import torch
from pyfaidx  import Fasta
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import pickle
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.state import AcceleratorState


class SequenceDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return idx, self.input_ids[idx], self.attention_mask[idx]

def create_dataloader(input_ids, attention_mask, batch_size):
    dataset = SequenceDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def main(args):
    fasta_dir = args.fasta_dir
    model_name = args.model_name
    output = args.output
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    #torch.set_printoptions(threshold=2000*2000*10)

    if 'facebook' in model_name:
        tokenizer, model = build_esm_model(model_name)
    else:
        tokenizer, model = None, None

    seqs = extract_seqs(fasta_dir)
    input_ids, attention_mask = generate_input_ids(tokenizer, seqs['seq'], max_seq_len)

    accelerator = Accelerator()

    eval_dataloader = create_dataloader(input_ids, attention_mask, batch_size)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    embeddings = generate_embeddings(model, eval_dataloader)
    all_embeddings = accelerator.gather(embeddings.contiguous())
    if accelerator.is_main_process:
        seqs['embeddings'] = all_embeddings.tolist()[:len(seqs['seq'])]
        data = pd.DataFrame(seqs)
        pickle.dump(data, open(output, 'wb'))

def generate_input_ids(tokenizer, seqs, max_seq_len):
    if max_seq_len is None:
        inputs = tokenizer.batch_encode_plus([str(seq) for seq in seqs],
                add_special_tokens=True, padding="longest", return_tensors="pt")
    else:
        inputs = tokenizer.batch_encode_plus([str(seq) for seq in seqs],
                add_special_tokens=True, padding="max_length", max_length=max_seq_len, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    return input_ids, attention_mask

def generate_embeddings(model, dataloader):
    all_embeddings = None
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            idx, input_ids, attention_mask = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs[0][:, 0, :]
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    return all_embeddings

def build_esm_model(model_name):
    from transformers import EsmTokenizer, EsmModel, EsmForMaskedLM
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    return tokenizer, model

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
