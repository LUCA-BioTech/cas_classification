import ankh
import torch
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from esm import FastaBatchedDataset, pretrained

class MyESM(object):

    def __init__(self, params):
        
        model, alphabet = pretrained.load_model_and_alphabet(params.embeding_model_name)
        self.model=model
        self.alphabet = alphabet
        self.model.eval()

        self.params=params

    def batch_iterable(self,iterable, batch_size):
        """Yields batches of a given size from an iterable."""
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]

    def extract(self,sequences_label,sequences):
        if self.params.cuda:
            self.model = self.model.cuda()
            
        dataset = FastaBatchedDataset(sequences_label,sequences)
        batches = dataset.get_batch_indices(self.params.embeding_tokens_per_batch, extra_toks_per_seq=1)

        data_loader = torch.utils.data.DataLoader(
            dataset, 
            collate_fn=self.alphabet.get_batch_converter(self.params.embeding_seq_length), 
            batch_sampler=batches
        )
        
        embed_array=[]
        label_array=[]

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):

                print(f'Processing batch {batch_idx + 1} of {len(batches)}')

                if self.params.cuda:
                    toks = toks.to(device="cuda", non_blocking=True)

                out = self.model(toks, repr_layers=[self.params.embeding_repr_layers_number], return_contacts=False)

                logits = out["logits"].to(device="cpu")
                representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
                
                for i, label in enumerate(labels):
                    # entry_id = str(label.split())[0]
                    # filename = output_dir / f"{entry_id}.pt"
                    truncate_len = min(self.params.embeding_seq_length, len(strs[i]))

                    embeding_data = {
                            layer: t[i, 1 : truncate_len + 1].mean(0).clone() for layer, t in representations.items()
                        }
                    
                    embed_array.append(embeding_data[self.params.embeding_repr_layers_number].numpy())
                    label_array.append(label)
                    
        return embed_array,label_array

