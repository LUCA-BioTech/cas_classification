import torch
from transformers import EsmModel, EsmTokenizer
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import numpy as np
import pandas as pd
import os
import joblib
import ast

def get_embeddings(sequences, model, tokenizer, max_length, device, batch_size=8):
    """
    Generate sequence embeddings by using the CLS token's embedding from the last hidden state.

    Args:
        sequences (List[str]): List of input sequences.
        model: Pretrained model used to generate embeddings.
        tokenizer: Tokenizer for the model.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        batch_size (int): Number of sequences to process in each batch.

    Returns:
        np.ndarray: Array of shape (num_sequences, hidden_size) containing embeddings.
    """
    num_sequences = len(sequences)          # Total number of sequences (S)
    all_embeddings = []                      # To store embeddings from all batches

    # Iterate over the sequences in batches
    for i in range(0, num_sequences, batch_size):
        batch_sequences = sequences[i:i+batch_size]  # Current batch of sequences (B,)

        # Tokenize the batch
        inputs = tokenizer(
            batch_sequences,
            return_tensors="pt",          # Return PyTorch tensors
            padding=True,                 # Pad sequences to the same length within the batch
            truncation=True,              # Truncate sequences longer than max_length
            max_length=max_length               # Maximum sequence length
        )
        # After tokenization:
        # inputs['input_ids'] shape: (B, L)
        # inputs['attention_mask'] shape: (B, L)

        # Move input tensors to the specified device (GPU or CPU)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():  # Disable gradient computation for efficiency
            outputs = model(**inputs)
            # outputs.hidden_states shape: (N, B, L, H)
            # N: Number of hidden layers in the model
            # B: Batch size
            # L: Sequence length
            # H: Hidden size

        # Extract the hidden states from all layers
        hidden_states = outputs.hidden_states           # Tuple of (N, B, L, H)
        last_hidden_state = hidden_states[-1]          # Last layer hidden state: (B, L, H)

        # The CLS token is always at position 0 in the sequence
        cls_embeddings = last_hidden_state[:, 0, :]  # (B, H) - Extract the CLS token's embedding

        # Move the embeddings to CPU and convert to NumPy array
        sequence_embeddings = cls_embeddings.cpu().numpy()  # (B, H)
        all_embeddings.append(sequence_embeddings)        # Append to the list

    # Concatenate all batch embeddings into a single NumPy array
    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (S, H)

    return all_embeddings

def load_cls_model(model_path):
    predictor = TabularPredictor.load(model_path)
    return predictor

def load_rd_transform(transform_path):
    transform  = joblib.load(transform_path)
    return transform

def reduce_dimension(embeddings, transform):
    embeddings_reduced = transform.transform(embeddings)
    return embeddings_reduced

def test(predictor, test_embeddings_reduced, test_labels=None):
    test_data = pd.DataFrame(test_embeddings_reduced)
    
    predictions = predictor.predict(test_data)

    if test_labels is not None:
        accuracy = (predictions == test_labels).mean()
        print(f'Accuracy: {accuracy:.4f}')
    
    return predictions

def read_fasta(file_path):
    fasta_dict = {}
    with open(file_path, 'r') as file:
        sequence_id = None
        sequence_lines = []
        for line in file:
            line = line.strip()
            if not line:
                continue 
            if line.startswith('>'):
                if sequence_id:
                    fasta_dict[sequence_id] = ''.join(sequence_lines)
                sequence_id = line[1:].split()[0]
                sequence_lines = []
            else:
                sequence_lines.append(line)
        if sequence_id:
            fasta_dict[sequence_id] = ''.join(sequence_lines)
    return fasta_dict

if __name__ == "__main__":

    model_name = "facebook/esm2_t33_650M_UR50D"   
    # Load the trained model from the checkpoint (make sure you point to the right directory)
    model = EsmModel.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    # Ensure the model is in evaluation mode
    model.eval()

    # Move the model to GPU if available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    cls_model_path = "./autogluon"
    rd_transform_path = "./pca/pca_model.pkl"

    cls_model = load_cls_model(cls_model_path)
    rd_transform = load_rd_transform(rd_transform_path)


#    fasta_file = "./test_data1.fasta"    # Need to put the filename and path here for tested sequence
#    sequence_list = []

#    seq_dict = read_fasta(fasta_file)
#    for seq_id, seq in seq_dict.items():
#        sequence_list.append(seq)

    # print(sequence_list)
#    embeddings = get_embeddings(sequence_list, model, tokenizer, max_length=1502, device=device)
    # print(embeddings.shape)
    new_test_path = './newtest.csv'
    new_test_df = pd.read_csv(new_test_path)
    embeddings = new_test_df['embeddings']  # List of protein sequences
    embeddings = [np.array(ast.literal_eval(item)) for item in embeddings]
    embeddings = np.array(embeddings)

    embeddings_reduced = rd_transform.transform(embeddings)
    predictions = test(cls_model, embeddings_reduced)
    for item1, item2 in zip(new_test_df['name'], predictions):
        print(item1, item2)

    
