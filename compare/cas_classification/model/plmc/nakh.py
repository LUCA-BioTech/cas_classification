import ankh
import torch
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

class Nakh(object):

    def __init__(self, params):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device }")

        # Load the selected model
        if params.embeding_model_name == 'large':
            self.model, self.tokenizer = ankh.load_large_model()
        elif params.embeding_model_name == 'base':
            self.model, self.tokenizer = ankh.load_base_model()
        else:
            raise ValueError("Invalid model type. Choose 'large' or 'base'.")

        self.model.to(self.device )
        self.model.eval()

        self.params=params

    def batch_iterable(self,iterable, batch_size):
        """Yields batches of a given size from an iterable."""
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]

    def extract(self,sequences_label,sequences):
        
        sequences_label = sequences_label.tolist()
        sequences = sequences.tolist()

        embed_array=[]
        label_array=[]

        for batch_seqs, batch_ids in zip(self.batch_iterable(sequences, self.params.batch_size), self.batch_iterable(sequences_label, self.params.batch_size)):
            ids = self.tokenizer.batch_encode_plus(batch_seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)

            last_hidden_states = embedding.last_hidden_state
            batch_embeddings = torch.mean(last_hidden_states, dim=1).cpu().numpy()

            for sequence_id, emb in zip(batch_ids, batch_embeddings):
                embed_array.append(emb)
                label_array.append(sequence_id)
                print(emb)
        
        return embed_array,label_array

