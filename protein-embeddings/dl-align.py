#!/usr/bin/env python
import argparse
import pickle
import numpy as np
import torch
from pyfaidx  import Fasta
from sentence_transformers import util

SCORE_SCALE_FACTOR = 100

def main(args):
    input_file1 = args.input_file1
    input_file2 = args.input_file2
    model_name = args.model_name
    embeddings_file = args.embeddings
    if input_file1 and input_file2 and model_name:
        sequence1 = str(Fasta(input_file1)[0])
        sequence2 = str(Fasta(input_file2)[0])
        calculate_seq_cos_sim(model_name, sequence1, sequence2)
    elif embeddings_file:
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        random_pair = args.random_pair
        name1 = args.name1
        name2 = args.name2
        if random_pair:
            category = args.category
            if category:
                data = data[data['category'] == category]
            for _ in range(random_pair):
                name1 = data['name'].sample(1).values[0]
                name2 = data['name'].sample(1).values[0]
                calculate_name_cos_sim(data, name1, name2)
        elif name1 and name2:
            calculate_name_cos_sim(data, name1, name2)

def get_row_data(data, name):
    row = data[data['name'] == name].iloc[0]
    embedding = row['embeddings']
    category = row['category']
    length = len(row['seq'])
    return embedding, category, length

def calculate_name_cos_sim(data, name1, name2):
    embedding1, category1, length1 = get_row_data(data, name1)
    embedding2, category2, length2 = get_row_data(data, name2)
    cosine_score, score = calculate_emb_similarity(embedding1, embedding2)
    percent = round(score * 100, 2)
    print("Similarity: {}%({}), {}/{} vs {}/{}, {} vs {}".format(percent, cosine_score, category1, length1, category2, length2, name1, name2))

def calculate_seq_cos_sim(model_name, sequence1, sequence2):
    if 'facebook' in model_name:
        tokenizer, model = build_esm_model(model_name)
    else:
        tokenizer, model = None, None
    ids = tokenizer.batch_encode_plus([sequence1, sequence2],
            add_special_tokens=True, padding="longest", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**ids)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        cosine_score, score = calculate_emb_similarity(embeddings[0], embeddings[1])
        print("Score: {}, Length are: {}, {}".format(score, len(sequence1), len(sequence2)))

def calculate_emb_similarity(embedding1, embedding2):
    cosine_score = util.cos_sim(embedding1, embedding2).item()
    factor = SCORE_SCALE_FACTOR
    score = np.exp(cosine_score * factor) / np.exp(factor)
    return cosine_score, score

def build_esm_model(model_name):
    from transformers import EsmTokenizer, EsmModel
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    return tokenizer, model


def create_parser():
    parser = argparse.ArgumentParser(description='Import sequence info from fasta files')
    parser.add_argument('-i1', '--input_file1', help='Input file 1')
    parser.add_argument('-i2', '--input_file2', help='Input file 2')
    parser.add_argument('--embeddings', help='Embeddings file')
    parser.add_argument('--name1', help='Name 1')
    parser.add_argument('--name2', help='Name 2')
    parser.add_argument('--random_pair', type=int, help='Random pair number')
    parser.add_argument('--category', help='Category')
    parser.add_argument('-m', '--model_name', help='model', default='facebook/esm2_t33_650M_UR50D')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
