#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
from nltk.cluster import KMeansClusterer, util
import pickle


def main(args):
    cluster_count = args.cluster_count
    input_file = args.input
    output_file = args.output
    data = pickle.load(open(input_file, 'rb'))
    embeddings = [np.array(embedding) for embedding in data['embeddings']]
    assigned_clusters = make_clusters(embeddings, cluster_count)
    data['cluster'] = pd.Series(assigned_clusters, index=data.index)
    data.to_csv(output_file, index=False)

def make_clusters(embeddings, cluster_count):
    kclusterer = KMeansClusterer(
            cluster_count, distance=util.euclidean_distance,
            repeats=25, avoid_empty_clusters=True)
    assigned_clusters = kclusterer.cluster(embeddings, assign_clusters=True)
    return assigned_clusters

def create_parser():
    parser = argparse.ArgumentParser(description='Embeddings clustering')
    parser.add_argument('-i', '--input', required=True, help='input embeddings')
    parser.add_argument('-o', '--output', required=True, help='output csv')
    parser.add_argument('-c', '--cluster_count', required=True, type=int, help='number of clusters')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
