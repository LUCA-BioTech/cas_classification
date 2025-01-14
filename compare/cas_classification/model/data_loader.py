import os
import random
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from esm import FastaBatchedDataset, pretrained
import utils 

from .plmc.my_esm import MyESM
from .plmc.nakh import Nakh
from .plmc.prot_t5 import Prot_t5

class DataLoader(object):

    def __init__(self, data_dir, params):
        self.params=params

    def load_seq_labels(self,csv_file,d):
        df = pd.read_csv(csv_file)

        mapping = {'Cas1': 1, 'Cas2': 2, 'Cas3': 3, 'Cas4': 4, 'Cas5': 5, 'Cas6': 6, 
                   'Cas7': 7, 'Cas8': 8, 'Cas9': 9, 'Cas10': 10, 'Cas12': 11, 'Cas13': 12,
                   'nocas': 0}
        
        df['label'] = df['label'].map(mapping)

        embed_file = os.path.join(os.path.dirname(csv_file),self.params.embeding_model_name+'_embed.npy')
        label_file = os.path.join(os.path.dirname(csv_file),self.params.embeding_model_name+'_label.npy')

        if os.path.exists(embed_file):
            print('load from embed file')
            seq_embed=np.load(embed_file)
            seq_lable=np.load(label_file)
        else:
            if self.params.embeding_model_name == 'base':
                nakh_model = Nakh(self.params)
                seq_embed,seq_lable=nakh_model.extract(sequences_label=df['label'],sequences=df['seq'])
            elif self.params.embeding_model_name == 'prot':
                prot_t5_model = Prot_t5(self.params)
                seq_embed,seq_lable=prot_t5_model.export(sequences_label=df['label'],sequences=df['seq'])
            else:
                esm_model = MyESM(self.params)
                seq_embed,seq_lable=esm_model.extract(sequences_label=df['label'],sequences=df['seq'])
            
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

                # print(f'{batch_labels}')
                # since all data are indices, we convert them to torch LongTensors
                batch_data_np = np.array(batch_data)
                batch_labels_np = np.array(batch_labels)

                batch_data, batch_labels = torch.FloatTensor(batch_data_np), torch.LongTensor(batch_labels_np)

                # shift tensors to GPU if available
                if params.cuda:
                    batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

                # convert them to Variables to record operations in the computational graph
                batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
        
                yield batch_data, batch_labels
