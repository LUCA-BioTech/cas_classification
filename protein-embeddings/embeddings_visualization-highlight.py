#!/usr/bin/env python
import argparse
import os
import plotly.express as px
import seaborn as sns
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm
from nltk.cluster import KMeansClusterer, util
import pandas as pd
import sys


def main(args):
    cluster_count = args.cluster_count
    output = args.output
    min_cas_length = args.min_cas_length
    cas_types = args.cas_types
    input_file = args.input
    highlight_input = args.highlight_input
    highlight_input2 = args.highlight_input2
    data = None
    highlight_data = None
    highlight_data2 = None
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    if highlight_input:
        with open(highlight_input, 'rb') as f:
            highlight_data = pickle.load(f)
    if highlight_input2:
        with open(highlight_input2, 'rb') as f:
            highlight_data2 = pickle.load(f)

    fig = visualize_clusters(data, cluster_count, min_cas_length, highlight_data=highlight_data, highlight_data2=highlight_data2, cas_types=cas_types)
    if output:
        with open(args.output, 'wb') as f:
            pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        plt.show()

def visualize_clusters(data, cluster_count, min_cas_length, highlight_data=None, highlight_data2=None, cas_types=None):
    print(highlight_data)
    highlight_categories = None
    highlight_categories2 = None
    if highlight_data is not None:
        data = pd.concat([data, highlight_data])
        highlight_categories = highlight_data['category'].unique()
    if highlight_data2 is not None:
        data = pd.concat([data, highlight_data2])
        highlight_categories2 = highlight_data2['category'].unique()
    data['length'] = data['seq'].apply(lambda x: len(x))
    if cas_types:
        data = data[data['category'].isin(cas_types)]
    if min_cas_length:
        data1 = data[~data['category'].isin(['Cas9', 'Cas12', 'Cas13'])]
        data2 = data[data['category'].isin(['Cas9', 'Cas12', 'Cas13'])]
        data = pd.concat([data1, data2[data2['length'] > min_cas_length]])
    # sort data with category
    data = data.sort_values(by=['category'])
    embeddings = data['embeddings']
    embeddings = np.array(embeddings.tolist())
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1500)
    tsne_results = tsne.fit_transform(embeddings)
    if cluster_count:
        assigned_clusters = make_clusters(tsne_results, cluster_count)
        data['cluster'] = pd.Series(assigned_clusters, index=data.index)
        labels = list(range(cluster_count))
        label_key = 'cluster'
    else:
        labels = data['category'].unique().tolist()
        labels.sort()
        label_key = 'category'

    data['x'] = tsne_results[:, 0]
    data['y'] = tsne_results[:, 1]

    cmap = cm.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]
    print(colors)
    colors = [f'rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})' for c in colors]
    fig = px.scatter(data, x='x', y='y', color='category', hover_data=['x', 'y', 'name', label_key, 'length'], color_discrete_sequence=colors)
    if highlight_categories is not None:
        for _, trace in enumerate(fig.data):
            if trace.name in highlight_categories:
                trace.marker.size = 8
                trace.marker.line.width = 1
                trace.marker.line.color = 'blue'
    if highlight_categories2 is not None:
        for _, trace in enumerate(fig.data):
            if trace.name in highlight_categories2:
                trace.marker.size = 8 
                trace.marker.line.width = 1
                trace.marker.line.color = 'red'
    fig.show()


def make_clusters(embeddings, cluster_count):
    kclusterer = KMeansClusterer(
            cluster_count, distance=util.euclidean_distance,
            repeats=25, avoid_empty_clusters=True)
    assigned_clusters = kclusterer.cluster(embeddings, assign_clusters=True)
    return assigned_clusters


def create_parser():
    parser = argparse.ArgumentParser(description='Embeddings visualization')
    parser.add_argument('-i', '--input', required=True, help='input embeddings')
    parser.add_argument('--highlight_input', help='input embeddings to highlight')
    parser.add_argument('--highlight_input2', help='input embeddings to highlight2')
    parser.add_argument('-o', '--output', help='dump figure path')
    parser.add_argument('-c', '--cluster_count', type=int, help='cluster count')
    parser.add_argument('--min_cas_length', type=int, help='minimum length for Cas9, Cas12, Cas13')
    parser.add_argument('--cas_types', nargs='+', help='cas types to use')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
