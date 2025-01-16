import torch
from transformers import EsmModel, EsmTokenizer
import numpy as np
import pandas as pd
import ast
import os
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt


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


def apply_pca(train_embeddings, test_embeddings, n_components):
    pca = PCA(n_components=n_components)
    train_embeddings_pca = pca.fit_transform(train_embeddings)
    test_embeddings_pca = pca.transform(test_embeddings)
    return train_embeddings_pca, test_embeddings_pca

# Function to apply UMAP for dimensionality reduction
def apply_umap(train_embeddings, test_embeddings, n_components, random_state=42):
    umap = UMAP(n_components=n_components, random_state=random_state)
    train_embeddings_umap = umap.fit_transform(train_embeddings)
    test_embeddings_umap = umap.transform(test_embeddings)
    return train_embeddings_umap, test_embeddings_umap

def apply_tsne(train_embeddings, test_embeddings, n_components):
    # Concatenate train and test embeddings
    combined_embeddings = np.vstack((train_embeddings, test_embeddings))
    
    # Apply t-SNE to the combined embeddings
    tsne = TSNE(n_components=n_components,perplexity=10)
    combined_embeddings_tsne = tsne.fit_transform(combined_embeddings)
    
    # Split back into train and test sets
    train_embeddings_tsne = combined_embeddings_tsne[:len(train_embeddings)]
    test_embeddings_tsne = combined_embeddings_tsne[len(train_embeddings):]
    
    return train_embeddings_tsne, test_embeddings_tsne


def evaluate(train_embeddings_reduced, train_labels, test_embeddings_reduced, test_labels, log_filename, model_save_dir):
    train_data = pd.DataFrame(train_embeddings_reduced)
    train_data['class'] = train_labels
    test_data = pd.DataFrame(test_embeddings_reduced)
    test_data['class'] = test_labels
    predictor = TabularPredictor(label='class', path=model_save_dir).fit(train_data, time_limit=120)  # Fit models for 120s

    # # Save the trained model with a unique name based on the number of components
    # model_filename = os.path.join(model_save_dir, f"model_{n_components}d_autogluon.pkl")
    # predictor.save(model_filename)
    
    # Evaluate the model on test data
    leaderboard = predictor.leaderboard(test_data)
    leaderboard.to_csv(log_filename, index=False)


def main(train_embeddings, test_embeddings, train_labels, test_labels, model_save_dir_base, log_save_dir, n_components=None, method="umap"):
    if n_components is not None: # n_components = 2
        if method == "umap":
            train_embeddings_reduced, test_embeddings_reduced = apply_umap(train_embeddings, test_embeddings, n_components=n_components)
        elif method == "pca":
            train_embeddings_reduced, test_embeddings_reduced = apply_pca(train_embeddings, test_embeddings, n_components=n_components)
        elif method == "tsne":
            train_embeddings_reduced, test_embeddings_reduced = apply_tsne(train_embeddings, test_embeddings, n_components=n_components)
    else:
        for n_components in range(2, 16):
            # Apply chosen dimensionality reduction method
            if method == "umap":
                train_embeddings_reduced, test_embeddings_reduced = apply_umap(train_embeddings, test_embeddings, n_components=n_components)
            elif method == "pca":
                train_embeddings_reduced, test_embeddings_reduced = apply_pca(train_embeddings, test_embeddings, n_components=n_components)
            elif method == "tsne" and n_components in (2, 3):
                train_embeddings_reduced, test_embeddings_reduced = apply_tsne(train_embeddings, test_embeddings, n_components=n_components)
            elif method == "raw":
                train_embeddings_reduced, test_embeddings_reduced = train_embeddings, test_embeddings
                if n_components == 3:
                    break
            else:
                raise ValueError("Invalid dimensionality reduction method. Choose 'umap' or 'pca' or 'tsne 2&3' or 'raw'.")
            
            if method == "raw":
                n_components = 1280
        # Log filename for each dimensionality setting
        log_filename = os.path.join(log_save_dir, f"results_{n_components}d.csv")
        model_save_dir = os.path.join(model_save_dir_base, f"{n_components}")
        
        # Evaluate and log results for each classifier, along with the best model summary
        evaluate(train_embeddings_reduced, train_labels, test_embeddings_reduced, test_labels, log_filename, model_save_dir)

def plot(embeddings, labels, save_path=None):
    tsne = TSNE(n_components=2, random_state=42,perplexity=10)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE of Protein Sequence Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":

    model_name = "output/models--facebook--esm2_t33_650M_UR50D/2024-12-02_00-47-40/epoch_0/"
    # Load the trained model from the checkpoint (make sure you point to the right directory)
    model = EsmModel.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    # Ensure the model is in evaluation mode
    model.eval()

    # Move the model to GPU if available
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    old_train_path = './Oldtrain.csv' 
    old_train_df = pd.read_csv(old_train_path)

    old_test_path = './Oldtest.csv' 
    old_test_df = pd.read_csv(old_test_path)

    train_embeddings = old_train_df['embedding'] 
    train_embeddings = [np.array(ast.literal_eval(item)) for item in train_embeddings]
    train_embeddings = np.array(train_embeddings)
    # print(train_embeddings.shape)

    train_labels = old_train_df['label']
    # print(train_labels.shape)

    test_embeddings = old_test_df['embedding']  
    test_embeddings = [np.array(ast.literal_eval(item)) for item in test_embeddings]
    test_embeddings = np.array(test_embeddings)
    # print(test_embeddings.shape)

    test_labels = old_test_df['label']
    # print(test_labels.shape)

    method = "umap"
    model_save_dir = f"./AutoglounModels/{method}"
    log_save_dir = f"./AutoglounLogs/{method}"
    os.makedirs(log_save_dir, exist_ok=True)
    main(train_embeddings, test_embeddings, train_labels, test_labels, model_save_dir, log_save_dir, method=method)

    plot(test_embeddings, test_labels)





