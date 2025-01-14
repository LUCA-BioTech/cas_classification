import argparse
import time
from pathlib import Path
import pandas as pd
import os

import torch
import time
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

# Create a models directory next to extract.py
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models")
os.makedirs(model_path, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Prot_t5(object):

    def __init__(self, params):
        self.params= params
    
    def get_T5_model(self,model_dir, transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"):
        print(f"Loading: {transformer_link}")
        if model_dir is not None:
            print("##########################")
            print(f"Loading cached model from: {model_dir}")
            print("##########################")
        model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
        if device == torch.device("cpu"):
            print("Casting model to full precision for running on CPU ...")
            model.to(torch.float32)

        model = model.to(device)
        model = model.eval()
        vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
        return model, vocab
    
    def export(self, sequences_label, sequences):
        """
        Embeds all sequences, grouped by their respective categories.
        :param sequences_label: List of labels corresponding to each sequence (categories).
        :param sequences: List of sequences.
        :return: Embedded values and their corresponding labels.
        """


        model, vocab = self.get_T5_model(self.params.embeding_model_name)

        max_residues = 4000
        max_seq_len = 2000
        max_batch = 50000

        # Group sequences by their labels into a dictionary
        seq_dict = {}
        for label, seq in zip(sequences_label, sequences):
            if label not in seq_dict:
                seq_dict[label] = []
            seq_dict[label].append(seq)

        avg_length = sum(len(seq) for seqs in seq_dict.values() for seq in seqs) / sum(len(seqs) for seqs in seq_dict.values())
        n_long = sum(1 for seqs in seq_dict.values() for seq in seqs if len(seq) > max_seq_len)

        print(f"Average sequence length: {avg_length}")
        print(f"Number of sequences >{max_seq_len}: {n_long}")

        start = time.time()
        emb_dict = {}

        for label, seqs in tqdm(seq_dict.items(), desc="Processing categories"):
            batch = []
            for seq_idx, seq in enumerate(tqdm(seqs, desc=f"Processing sequences in category {label}"), 1):
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seq_len = len(seq)
                seq = ' '.join(list(seq))
                batch.append((label, seq, seq_len))

                n_res_batch = sum(s_len for _, _, s_len in batch) + seq_len
                if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seqs) or seq_len > max_seq_len:
                    labels, seqs, seq_lens = zip(*batch)
                    batch = []

                    token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                    input_ids = torch.tensor(token_encoding['input_ids']).to(device)
                    attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

                    try:
                        with torch.no_grad():
                            embedding_repr = model(input_ids, attention_mask=attention_mask)
                    except RuntimeError:
                        print(f"RuntimeError during embedding for {label} (L={seq_len}). Try lowering batch size. " +
                              "If single sequence processing does not work, you need more vRAM to process your protein.")
                        continue

                    for batch_idx, identifier in enumerate(labels):
                        s_len = seq_lens[batch_idx]
                        emb = embedding_repr.last_hidden_state[batch_idx, :s_len]

                        # Aggregate embeddings if required
                        emb = emb.mean(dim=0)

                        if identifier not in emb_dict:
                            emb_dict[identifier] = []
                        emb_dict[identifier].append(emb.detach().cpu().numpy().squeeze())

        end = time.time()

        # Print statistics
        print('\n############# STATS #############')
        print(f'Total number of embeddings: {sum(len(v) for v in emb_dict.values())}')
        print(f'Total time: {end - start:.2f}[s]; time/prot: {(end - start) / sum(len(v) for v in emb_dict.values()):.4f}[s]; avg. len= {avg_length:.2f}')

        # Flatten embeddings for output
        flattened_embs = []
        flattened_labels = []
        for label, embeddings in emb_dict.items():
            for emb in embeddings:
                flattened_embs.append(emb)
                flattened_labels.append(label)

        return flattened_embs, flattened_labels

    

