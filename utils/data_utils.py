import os
import re
import pyfaidx
import yaml
from sklearn.preprocessing import MultiLabelBinarizer
import utils.constants as constants

def load_yaml(path):
    with open(path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

def get_label_convertion_fn(all_labels):
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb = mlb.fit(all_labels)
    print(f"MultiLabel Mapping: {all_labels}, {mlb.transform([[i] for i in all_labels])}")
    def label_to_id_fn(labels, is_multi_label):
        if is_multi_label:
            return mlb.transform(labels)
        else:
            return [all_labels.index(label) if label in all_labels else 0 for label in labels]
    def id_to_label_fn(ids, is_multi_label):
        if is_multi_label:
            return mlb.inverse_transform(ids)
        else:
            return [all_labels[id] for id in ids]
    return label_to_id_fn, id_to_label_fn

def read_seq_labels_and_metadata(config):
    if not config or not config.file:
        return None, None
    category_weights = config.category_weights
    seq_labels = {}
    weight_category_index = None
    domain_pos_index = None
    max_num_seq_labels = 1
    lines = open(config.file, 'r').readlines()
    name_index = config.index.name
    labels_index = config.index.labels
    weight_category_index = config.index.weight_category
    weight_index = config.index.weight
    domain_pos_index = config.index.domain_position
    metadata = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_splits = line.split('\t')
        name = line_splits[name_index]

        label_str = line_splits[labels_index]
        if not label_str or label_str.lower() == constants.NONE_LABEL_PLACEHOLDER.lower():
            seq_labels[name] = []
            num_seq_labels = 0
        else:
            cur_seq_labels = label_str.split(',')
            seq_labels[name] = cur_seq_labels
            num_seq_labels = len(cur_seq_labels)
        if num_seq_labels > max_num_seq_labels:
            max_num_seq_labels = num_seq_labels

        seq_meta = {}
        if weight_category_index is not None:
            seq_meta[constants.WEIGHTS_SHORT_KEY] = category_weights[line_splits[weight_category_index]]
        if weight_index is not None:
            seq_meta[constants.WEIGHTS_SHORT_KEY] = line_splits[weight_index]
        if domain_pos_index is not None:
            seq_meta[constants.POSITIONS_SHORT_KEY] = [int(i) for i in line_splits[domain_pos_index].split(',')]
        if seq_meta:
            metadata[name] = seq_meta
    if max_num_seq_labels == 1:
        for name in seq_labels:
            seq_labels[name] = seq_labels[name][0]
    return seq_labels, metadata

def read_fasta(fasta_dir, seq_labels=None, seq_metadata=None):
    labels = []
    names = []
    sequences = []
    metadata = []

    for fasta_file in os.listdir(fasta_dir):
        if not fasta_file.endswith(('.faa', '.fasta')):
            continue
        fasta = pyfaidx.Fasta(os.path.join(fasta_dir, fasta_file), rebuild=False)
        file_basename = fasta_file.split('.')[0]
        for record in fasta:
            name = record.name
            if seq_labels is not None:
                label = seq_labels[name]
                if seq_metadata is not None:
                    metadata.append(seq_metadata[name])
            else:
                label = file_basename
            labels.append(label) # type: ignore
            seq = str(record)
            seq = re.sub(r"[\n\*]", '', seq)
            seq = re.sub(r"[UZOB]", "X", seq)
            sequences.append(seq)
            names.append(name)

    print(f"Read {len(labels)} sequences from {fasta_dir}, sequences: {len(sequences)}, names: {len(names)} from fasta_dir: {fasta_dir}")
    return labels, sequences, names, metadata

def read_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels

def set_domain_positions(inputs, batch, domain_aware):
    if "metadata" in batch:
        metadata = batch["metadata"]
        pos_key = constants.POSITIONS_SHORT_KEY
        if pos_key in metadata:
            positions = metadata[pos_key]
            inputs["start_positions"] = positions[0]
            inputs["end_positions"] = positions[1]
            if domain_aware:
                inputs["domain_classification_strategy"] = domain_aware.classification_strategy

def get_sample_weights(batch):
    if "metadata" in batch:
        metadata = batch["metadata"]
        return metadata.get(constants.WEIGHTS_SHORT_KEY)

