import os
import random
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from esm import FastaBatchedDataset, pretrained
import utils 

class DataLoader(object):

    def __init__(self, data_dir, params):
        model, alphabet = pretrained.load_model_and_alphabet(params.embeding_model_name)
        self.model=model
        self.alphabet = alphabet
        self.model.eval()

        self.params=params

    def extract_embeddings(self,labels,seq):
        
        if self.params.cuda:
            self.model = self.model.cuda()
            
        dataset = FastaBatchedDataset(labels,seq)
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

    def load_seq_labels(self,csv_file,d):
        df = pd.read_csv(csv_file)

        embed_file = os.path.join(os.path.dirname(csv_file),self.params.embeding_model_name+'_embed.npy')
        label_file = os.path.join(os.path.dirname(csv_file),self.params.embeding_model_name+'_label.npy')
        if os.path.exists(embed_file):
            print('load from embed file')
            seq_embed=np.load(embed_file)
            seq_lable=np.load(label_file)
        else:
            seq_embed,seq_lable=self.extract_embeddings(df['label'],df['seq'])
            np.save(embed_file, seq_embed)
            np.save(label_file, seq_lable)
            print('wite embed file')

        d['data'] = seq_embed
        d['labels'] = seq_lable
        # d['id'] = df['id']
        d['size'] = len(seq_embed)

    def load_data(self, types, data_dir):
        """
        Loads the data for each type in types from data_dir.

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
            data_dir: (string) directory containing the dataset

        Returns:
            data: (dict) contains the data with labels for each type in types

        """
        data = {}
        
        for split in ['train', 'val', 'test']:
            if split in types:
                seq_file = os.path.join(data_dir, split, split+".csv")
                data[split] = {}
                self.load_seq_labels(seq_file, data[split])

        return data
    
    def data_iterator(self, data, params, shuffle=False):
            """
            Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
            pass over the data.

            Args:
                data: (dict) contains data which has keys 'data', 'labels' and 'size'
                params: (Params) hyperparameters of the training process.
                shuffle: (bool) whether the data should be shuffled

            Yields:
                batch_data: (Variable) dimension batch_size x seq_len with the sentence data
                batch_labels: (Variable) dimension batch_size x seq_len with the corresponding labels

            """

            # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
            order = list(range(data['size']))
            if shuffle:
                random.seed(230)
                random.shuffle(order)

            # one pass over data
            for i in range((data['size']+1)//params.batch_size):
                # fetch sentences and tags
                batch_data = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
                batch_labels = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]

                # since all data are indices, we convert them to torch LongTensors
                batch_data, batch_labels = torch.FloatTensor(batch_data), torch.LongTensor(batch_labels)

                # shift tensors to GPU if available
                if params.cuda:
                    batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

                # convert them to Variables to record operations in the computational graph
                batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
        
                yield batch_data, batch_labels
