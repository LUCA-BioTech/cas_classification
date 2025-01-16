import argparse
from datetime import datetime
import json
import time
import yaml
import logging
import os
import re
import ast

from accelerate import Accelerator
from accelerate.utils import DummyScheduler, DummyOptim, set_seed
import pandas as pd
import numpy as np
import pyfaidx
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel, EsmForSequenceClassification
from transformers.trainer_pt_utils import get_parameter_names
import deepspeed
from deepspeed.accelerator import get_accelerator
import wandb

from utils.focalloss import FocalLoss
from utils.attention_utils import get_masked_attention, get_batch_head_attention, get_batch_head_sequence_attention
import shutil

wandb_run = None

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record) 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())

DATASET_TRAINING_KEYS = ['labels', 'input_ids', 'attention_mask']

def create_parser():
    parser = argparse.ArgumentParser(description='ESM2 Classification')
    parser.add_argument('--train_dataset_dir', type=str, help='train datasets directory that includes FASTA files named according to their labels')
    parser.add_argument('--eval_dataset_dir', type=str, required=True, help='eval datasets directory that includes FASTA files named according to their labels')
    parser.add_argument('--label_file', type=str, help='label file for train/eval/predict')
    parser.add_argument('-a', '--action', type=str, required=True, help='action', choices=["train", "eval", "predict", "export"])
    parser.add_argument('--attention_strategy', type=str, help='calculation strategy for attention export', default=None, choices=["full-huge-storage-to-layer", "full-huge-storage-to-head", "2d", "3d-max", "3d-average", "3d-high_confidence_average"])
    parser.add_argument('--attention_layers', nargs='+', type=str, help='layers for attention export, use "all" for all layers', default=[-3, -2, -1])
    parser.add_argument('--attention_threshold', type=float, help='threshold for attention calculation')
    parser.add_argument('--predict_head_mask', type=parse_head_mask, help='Mask to nullify selected heads of the self-attention modules. Mask values selected in [0, 1]: 1 indicates the head is not masked, 0 indicates the head is masked. examples: --predict_head_mask  "[0, 1, ... all heads ..., 1, 0]"  or --predict_head_mask "[[0, 1, ... all heads ..., 1, 0], ... all layers ..., [1, 0, ... all heads ..., 0, 1]]"')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='model name')
    parser.add_argument('-l', '--max_seq_len', type=int, required=True, help='max sequence length')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory')
    parser.add_argument('--num_train_epochs', type=int, help='number of epochs', default=1)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--min_learning_rate', type=float, help='min learning rate', default=0)
    parser.add_argument('--max_train_steps', type=int, help='max_train_steps', default=None)
    parser.add_argument('--micro_train_batch_size', type=int, help='train batch size per device', default=2)
    parser.add_argument('--eval_batch_size', type=int, help='eval batch size per device', default=10)
    parser.add_argument('--random_seed', type=int, help='random seed', default=42)
    parser.add_argument('--gradient_checkpointing', action='store_true', help='gradient checkpointing')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='gradient accumulation steps', default=1)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=0.01)
    parser.add_argument('--save_steps', type=int, help='save_steps', default=-1)
    parser.add_argument('--num_warmup_steps', type=int, help='warmup steps', default=1000)
    parser.add_argument('--logging_steps', type=int, help='logging_steps', default=100)
    parser.add_argument('--logging_dir', type=str, help='logging_dir', default="./logs")
    parser.add_argument('--output_file', type=str, help='predict/export output file')
    parser.add_argument('--checkpoint', type=str, help="checkpoint for resume")
    parser.add_argument('--loss_fn', type=str, help="Loss function", choices=["focal"])
    parser.add_argument("--project", type=str, help="W&B project name", default="esm2_classification")
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    return parser

def parse_head_mask(value):
    try:
        print(torch.tensor(ast.literal_eval(value)))
        return torch.tensor(ast.literal_eval(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid head_mask value: {value}")

def read_config(path):
    with open(path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

class SequenceDataset(Dataset):
    def __init__(self, inputs, labels, names):
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.labels = torch.tensor(labels)
        self.names = names
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {'labels': self.labels[idx], 'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'ids': idx}
    def get_num_samples_per_class(self):
        return torch.bincount(self.labels).tolist()

def create_dataset(tokenizer, fasta_dir, max_seq_len, label_to_id_fn, random_seed):
    labels, sequences, names = read_fasta(fasta_dir)
    if random_seed is not None:
        labels, sequences, names = shuffle(labels, sequences, names, random_state=random_seed) # type: ignore
    inputs = tokenizer(sequences, padding=True, truncation=True, max_length=max_seq_len, return_tensors='pt', add_special_tokens=True)
    label_ids = label_to_id_fn(labels)
    return SequenceDataset(inputs, label_ids, names)

def compute_metrics(labels, logits, all_labels):
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=list(range(len(all_labels))), average=None)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()}

def read_fasta(fasta_dir):
    labels = []
    names = []
    sequences = []
    for fasta_file in os.listdir(fasta_dir):
        if not fasta_file.endswith(('.faa', '.fasta')):
            continue
        label = fasta_file.split('.')[0]
        fasta = pyfaidx.Fasta(os.path.join(fasta_dir, fasta_file), rebuild=False)
        for record in fasta:
            labels.append(label)
            seq = str(record)
            seq = re.sub(r"[\n\*]", '', seq)
            seq = re.sub(r"[UZOB]", "X", seq)
            sequences.append(seq)
            names.append(record.name)
    print(f"Read {len(labels)} sequences from {fasta_dir}, sequences: {len(sequences)}, names: {len(names)} from fasta_dir: {fasta_dir}")
    time.sleep(1) # avoid multi process issues
    return labels, sequences, names

def get_dataloader(tokenizer, label_to_id_fn, random_seed, args):
    if args.do_train:
        train_dataset = create_dataset(tokenizer, args.train_dataset_dir, args.max_seq_len, label_to_id_fn, random_seed)
        train_dataloader = DataLoader(train_dataset, batch_size=args.micro_train_batch_size)
    else:
        train_dataloader = None
        train_dataset = None
    eval_dataset = create_dataset(tokenizer, args.eval_dataset_dir, args.max_seq_len, label_to_id_fn, random_seed)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader, train_dataset, eval_dataset

def get_optimizer(model, accelerator, args):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": args.weight_decay,
                },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
                },
            ]

    optimizer_cls = (
        AdamW
        if "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    return optimizer

def make_scheduler(args, accelerator, train_dataloader, optimizer):
    warmup_steps = args.num_warmup_steps
    lr_ratio = args.min_learning_rate / args.learning_rate
    accelerator.print(f"Len of train_dataloader: {len(train_dataloader)}")
    total_num_steps = (len(train_dataloader) / accelerator.gradient_accumulation_steps) * args.num_train_epochs
    total_num_steps += int(total_num_steps * lr_ratio) + warmup_steps
    accelerator.print(f"Total training steps: {total_num_steps}")
    scheduler = DummyScheduler(
        optimizer, total_num_steps=total_num_steps, warmup_num_steps=warmup_steps
    )
    return scheduler


def compute_class_weights(num_samples_per_class):
    epsilon=1e-6
    weights = [1 / (n + epsilon) for n in num_samples_per_class]
    normalized_weights = [w / sum(weights) for w in weights]
    return torch.tensor(normalized_weights)

def get_focal_loss_fn(train_dataset, eval_dataset):
    if train_dataset:
        num_samples_per_class = train_dataset.get_num_samples_per_class()
    else:
        num_samples_per_class = eval_dataset.get_num_samples_per_class()
    logger.info(f"num_samples_per_class: {num_samples_per_class}")
    #print(f"num_samples_per_class: {num_samples_per_class}")
    alpha = compute_class_weights(num_samples_per_class)
    gamma = 2

    loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    return loss_fn


def read_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels

def get_label_to_id_fn(all_labels):
    def label_to_id_fn(labels):
        return [all_labels.index(label) if label in all_labels else 0 for label in labels]
    return label_to_id_fn

def get_loss_logits(outputs, labels, loss_fn):
    logits = outputs.get("logits")
    if loss_fn:
        loss = loss_fn(logits, labels)
    else:
        loss = outputs.get("loss")
    return loss, logits

def do_train(model, train_dataloader, loss_fn, optimizer, scheduler, args, epoch, accelerator):
    global wandb_run
    model.train()
    log_steps = args.logging_steps
    mean_loss = MeanMetric(nan_strategy="error").to(accelerator.device)
    length = len(train_dataloader)
    for step, batch in enumerate(tqdm(train_dataloader), start=1):
        inputs = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
        labels = batch.get("labels")
        outputs = model(**inputs)
        loss, _ = get_loss_logits(outputs, labels, loss_fn)
        gathered_loss = accelerator.gather_for_metrics((loss))
        mean_loss.update(gathered_loss)
        get_accelerator().empty_cache()
        while True:
            try:
                accelerator.backward(loss)
                break
            except torch.cuda.OutOfMemoryError: # type: ignore
                # Occasionally, there may arise a dearth of memory.
                logger.warning("CUDA out of memory. Retrying the current batch.")
                get_accelerator().empty_cache()
                time.sleep(1)
        if step % accelerator.gradient_accumulation_steps == 0 or step == length:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        global_step = step + epoch * length
        train_loss = mean_loss.compute().item()
        lr = optimizer.param_groups[0]['lr']
        if global_step % log_steps == 0:
            accelerator.print(f"\nepoch: {epoch}: step: {global_step}, loss: {train_loss}, lr: {lr}")
        if accelerator.is_main_process:
            if wandb_run:
                wandb_run.log({ "train_loss": train_loss, "lr": lr, "global_step": global_step })
        if args.save_steps > 0 and global_step % args.save_steps == 0:
            accelerator.save_state(f"{args.output_dir}/step_{global_step}")
            previous_save_step = global_step - args.save_steps
            if previous_save_step > 0:
                accelerator.wait_for_everyone()
                accelerator.print(f"Deleting step: {previous_save_step}")
                shutil.rmtree(f"{args.output_dir}/step_{previous_save_step}", ignore_errors=True)


def do_eval(model, eval_dataloader, loss_fn, all_labels, accelerator, epoch=None):
    global wandb_run
    model.eval()
    merged_labels = torch.tensor([], dtype=torch.int)
    merged_logits = torch.tensor([], dtype=torch.float)
    mean_loss = MeanMetric(nan_strategy="error").to(accelerator.device)
    with torch.no_grad():
        eval_loss = 0
        for batch in tqdm(eval_dataloader):
            inputs = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
            labels = batch.get("labels")
            outputs = model(**inputs)
            loss, logits = get_loss_logits(outputs, labels, loss_fn)
            gathered_loss = accelerator.gather_for_metrics((loss))
            mean_loss.update(gathered_loss)
            gathered_labels, gathered_logits = accelerator.gather((labels, logits))
            if accelerator.is_main_process:
                merged_labels = torch.cat((merged_labels, gathered_labels.cpu()), dim=0)
                merged_logits = torch.cat((merged_logits, gathered_logits.cpu()), dim=0)
    eval_loss = mean_loss.compute().item()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metrics = compute_metrics(merged_labels, merged_logits, all_labels)
        eval_info =f"eval_loss: {eval_loss} eval metrics: {metrics}"
        if epoch:
            eval_info = f"Epoch {epoch} - {eval_info}"
        accelerator.print(eval_info)
        if wandb_run:
            wandb_run.log({ "eval_loss": eval_loss, **metrics })

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
def do_predict(model, head_mask, eval_dataloader, all_labels, output_file, accelerator):
    model.eval()
    merged_ids = torch.tensor([], dtype=torch.int)
    merged_label_ids = torch.tensor([], dtype=torch.int)
    merged_predicted_label_ids = torch.tensor([], dtype=torch.int)
    merged_logits = torch.tensor([], dtype=torch.float)
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            inputs = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
            label_ids = batch.get("labels")
            ids = batch.get("ids")
            if head_mask:
                inputs["head_mask"] = head_mask
            outputs = model(**inputs)
            logits = outputs.get("logits")
            predicted_probs = torch.softmax(logits, dim=1)
            predicted_label_ids = torch.argmax(logits, dim=1)
            gathered_ids, gathered_label_ids, gathered_predicted_label_ids, gathered_predicted_probs = accelerator.gather((ids, label_ids, predicted_label_ids, predicted_probs))
            if accelerator.is_main_process:
                merged_ids = torch.cat((merged_ids, gathered_ids.cpu()), dim=0)
                merged_label_ids = torch.cat((merged_label_ids, gathered_label_ids.cpu()), dim=0)
                merged_predicted_label_ids = torch.cat((merged_predicted_label_ids, gathered_predicted_label_ids.cpu()), dim=0)
                merged_logits = torch.cat((merged_logits, gathered_predicted_probs.cpu()), dim=0)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged_labels = [all_labels[label_id] for label_id in merged_label_ids]
        merged_predicted_labels = [all_labels[label_id] for label_id in merged_predicted_label_ids]
        merged_names = [eval_dataloader.dataset.names[id] for id in merged_ids]
        
        # 计算混淆矩阵、精确率、召回率、F1分数
        y_true = merged_label_ids.numpy()
        y_pred = merged_predicted_label_ids.numpy()
        conf_matrix = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print("Confusion Matrix:")
        print(conf_matrix)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        
        df = pd.DataFrame({
            "name": merged_names,
            "label": merged_labels,
            "predicted_label": merged_predicted_labels,
        })
        for i, label in enumerate(all_labels):
            df[f"prob: {label}"] = [f"{round(prob[i] * 100, 2)}%" for prob in merged_logits.numpy()]
        df.to_csv(output_file, index=False)

def do_predict_tmp(model, head_mask, eval_dataloader, all_labels, output_file, accelerator):
    model.eval()
    merged_ids = torch.tensor([], dtype=torch.int)
    merged_label_ids = torch.tensor([], dtype=torch.int)
    merged_predicted_label_ids = torch.tensor([], dtype=torch.int)
    merged_logits = torch.tensor([], dtype=torch.float)
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            inputs = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
            label_ids = batch.get("labels")
            ids = batch.get("ids")
            if head_mask:
                inputs["head_mask"] = head_mask
            outputs = model(**inputs)
            logits = outputs.get("logits")
            predicted_probs = torch.softmax(logits, dim=1)
            predicted_label_ids = torch.argmax(logits, dim=1)
            gathered_ids, gathered_label_ids, gathered_predicted_label_ids, gathered_predicted_probs = accelerator.gather((ids, label_ids, predicted_label_ids, predicted_probs))
            if accelerator.is_main_process:
                merged_ids = torch.cat((merged_ids, gathered_ids.cpu()), dim=0)
                merged_label_ids = torch.cat((merged_label_ids, gathered_label_ids.cpu()), dim=0)
                merged_predicted_label_ids = torch.cat((merged_predicted_label_ids, gathered_predicted_label_ids.cpu()), dim=0)
                merged_logits = torch.cat((merged_logits, gathered_predicted_probs.cpu()), dim=0)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged_labels = [all_labels[label_id] for label_id in merged_label_ids]
        merged_predicted_labels = [all_labels[label_id] for label_id in merged_predicted_label_ids]
        merged_names = [eval_dataloader.dataset.names[id] for id in merged_ids]
        accelerator.print(f"merged_names: len: {len(merged_names)}, merged_labels: len: {len(merged_labels)}, merged_predicted_labels: len: {len(merged_predicted_labels)}, merged_logits: len: {len(merged_logits)}")
        df = pd.DataFrame({
            "name": merged_names,
            "label": merged_labels,
            "predicted_label": merged_predicted_labels,
        })
        for i, label in enumerate(all_labels):
            df[f"prob: {label}"] = [f"{round(prob[i] * 100, 2)}%" for prob in merged_logits.numpy()]
        df.to_csv(output_file, index=False)

def do_export(model, eval_dataloader, output_dir, accelerator, attention_strategy, attention_threshold, attention_layers):
    model.eval()
    merged_ids = torch.tensor([], dtype=torch.int)
    num_seqs = eval_dataloader.dataset.__len__()
    #merged_residue_representations = torch.tensor([], dtype=torch.float)
    merged_embeddings = torch.tensor([], dtype=torch.float)
    unwrapped_model = accelerator.unwrap_model(model)
    num_layers = len(unwrapped_model.encoder.layer)
    accelerator.print(f"The model has {num_layers} layers")
    merged_aggregated_attentions = [torch.tensor([], dtype=torch.float).cpu() for _ in range(num_layers)]

    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            inputs = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS and k != "labels"}
            ids = batch.get("ids")
            names = [eval_dataloader.dataset.names[id] for id in ids]
            outputs = model(output_attentions=True, **inputs)
            #residue_representation = outputs.get("last_hidden_state")
            embedding = outputs.get("pooler_output")
            #gathered_ids, gathered_residue_representations, gathered_embeddings = accelerator.gather((ids, residue_representation, embedding))
            gathered_ids, gathered_embeddings = accelerator.gather((ids, embedding))
            gathered_names = [eval_dataloader.dataset.names[id] for id in gathered_ids]
            if accelerator.is_main_process:
                merged_ids = torch.cat((merged_ids, gathered_ids.cpu()), dim=0)
                #merged_residue_representations = torch.cat((merged_residue_representations, gathered_residue_representations.cpu()), dim=0)
                merged_embeddings = torch.cat((merged_embeddings, gathered_embeddings.cpu()), dim=0)
            if attention_strategy:
                attentions = outputs.get("attentions")
                attention_mask = batch.get("attention_mask")
                gathered_attention_mask = accelerator.gather(attention_mask)
                merged_names = [eval_dataloader.dataset.names[id] for id in merged_ids]
                # handle attention for each layer
                layers_attentions = get_layers_attentions(attentions, attention_layers)
                for i, attention in enumerate(layers_attentions):
                    aggregated_attention = None
                    if attention_strategy.startswith("full-huge-storage"):
                        gathered_attention = accelerator.gather(attention)
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            store_to_level = attention_strategy.split("-")[-1]
                            save_attention(i, gathered_attention.cpu()[:num_seqs], gathered_names[:num_seqs], gathered_attention_mask[:num_seqs], store_to_level, output_dir)
                    elif attention_strategy == "2d":
                        aggregated_attention = get_batch_head_attention(attention, attention_threshold, attention_mask)
                    elif attention_strategy.startswith("3d-"):
                        aggregated_attention = get_batch_head_sequence_attention(attention, attention_threshold, attention_strategy.split("-")[1], attention_mask)
                    if aggregated_attention is not None:
                        gathered_aggregated_attention = accelerator.gather((aggregated_attention))
                        if accelerator.is_main_process:
                            merged_aggregated_attentions[i] = torch.cat((merged_aggregated_attentions[i], gathered_aggregated_attention.cpu()), dim=0)

    if accelerator.is_main_process:
        merged_aggregated_attentions = torch.stack(merged_aggregated_attentions).transpose(0,1)
        merged_names = [eval_dataloader.dataset.names[id] for id in merged_ids]
        accelerator.print(f"merged_names: {merged_names}")
        accelerator.print(f"merged_names: {len(merged_names)}, merged_embeddings: {merged_embeddings.shape}, merged_aggregated_attentions: {merged_aggregated_attentions.shape}")
        df_data = {
            "id": merged_ids.numpy().tolist(),
            "name": merged_names,
            #"residue_representation": merged_residue_representations.numpy().tolist(),
            "embedding": merged_embeddings.numpy().tolist(),
        }
        if attention_strategy is not None and not attention_strategy.startswith("full-huge-storage"):
            df_data["aggregated_attention"] = merged_aggregated_attentions.numpy().tolist()
        df = pd.DataFrame(df_data).head(num_seqs)
        df.to_csv(f"{output_dir}/export-{attention_strategy}-{attention_threshold}.csv", index=False)

def get_layers_attentions(attentions, layers):
    if layers[0] == "all":
        return attentions
    else:
        return tuple([attentions[int(i)] for i in layers])

def save_attention(layer_index, attention, names, attention_masks, store_to_level, output_dir):
    for i, head_attentions in enumerate(attention):
        name = names[i]
        if not os.path.exists(os.path.join(output_dir,name)):
            os.mkdir(os.path.join(output_dir,name))
        attention_mask = attention_masks[i]
        masked_head_attentions = get_masked_attention(head_attentions, attention_mask)


        if store_to_level == 'layer':
            file_path = f"{output_dir}/{name}/attention_{name}_{layer_index}.jsonl"
            with open(file_path, "w") as f:
                f.write(json.dumps([t.tolist() for t in masked_head_attentions]) + "\n")
        elif store_to_level == 'head':
            for j in range(len(masked_head_attentions)):
                file_path = f"{output_dir}/{name}/attention_{name}_{layer_index}-{j}.jsonl"
                with open(file_path, "w") as f:
                    f.write(json.dumps([t.tolist() for t in masked_head_attentions[j]]) + "\n")
        else:
            raise RuntimeError(f"Wrong store to level: {store_to_level}")

def process_action(args):
    args.do_train = False
    args.do_eval = False
    args.do_predict = False
    args.do_export = False
    if args.action == "train":
        assert args.train_dataset_dir is not None
        assert args.output_dir is not None
        assert args.label_file is not None
        assert args.output_file is None
        args.output_dir = f"{args.output_dir}/{args.model_name.split('/')[-1]}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        args.do_train = True
    elif args.action == "eval":
        assert args.eval_dataset_dir is not None
        assert args.label_file is not None
        assert args.output_file is None
        assert args.output_dir is None
        args.do_eval = True
    elif args.action == "predict":
        assert args.eval_dataset_dir is not None
        assert args.output_file is not None
        assert args.label_file is not None
        assert args.output_dir is None
        args.do_predict = True
    elif args.action == "export":
        assert args.eval_dataset_dir is not None
        assert args.output_file is None
        assert args.output_dir is not None
        args.do_export = True
        if args.attention_strategy and "full" not in args.attention_strategy:
            assert args.attention_threshold is not None
    else:
        pass

def process_training(model, tokenizer, train_dataloader, eval_dataloader, loss_fn, all_labels, accelerator, args):

    optimizer = get_optimizer(model, accelerator, args)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    scheduler = make_scheduler(args, accelerator, train_dataloader, optimizer)

    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    accelerator.register_for_checkpointing(scheduler)
    if args.checkpoint:
        accelerator.load_state(args.checkpoint)
        accelerator.print(f"Resumed from checkpoint: {args.checkpoint}")
        resume_step = int(args.checkpoint.split("_")[-1])
        accelerator.skip_first_batches(train_dataloader, resume_step)
        accelerator.print(f"Resuming from step {resume_step}")

    os.makedirs(args.output_dir, exist_ok=True)

    global wandb_run
    if accelerator.is_main_process:
        wandb_run = wandb.init(config=args, project=args.project, resume="allow")

    for epoch in range(args.num_train_epochs):
        epoch_save_dir = f"{args.output_dir}/epoch_{epoch}"
        
        do_train(model, train_dataloader, loss_fn, optimizer, scheduler, args, epoch, accelerator)
        
        accelerator.wait_for_everyone()
        
        do_eval(model, eval_dataloader, loss_fn, all_labels, accelerator, epoch)
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            epoch_save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(epoch_save_dir)
        accelerator.print(f"Saved model checkpoint to {args.output_dir}/epoch_{epoch}")
    with open(f"{args.output_dir}/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    with open(f"{args.output_dir}/deepspeed.json", "w") as f:
        json.dump(accelerator.state.deepspeed_plugin.deepspeed_config, f, indent=2)

def make_accelerator(args):
    accelerator = Accelerator()
    deepspeed.init_distributed()
    deepspeed_config = accelerator.state.deepspeed_plugin.deepspeed_config # type: ignore
    if args.gradient_accumulation_steps > 1:
        deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    else:
        args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
    accelerator.print(args)
    accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
    deepspeed_config["train_micro_batch_size_per_gpu"] = args.micro_train_batch_size
    accelerator.print(f"train_micro_batch_size_per_gpu: {args.micro_train_batch_size}")
    deepspeed_config["train_batch_size"] = args.micro_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print(deepspeed_config)
    return accelerator

def main(args):
    process_action(args)

    random_seed = args.random_seed
    set_seed(random_seed)

    accelerator = make_accelerator(args)

    label_file = args.label_file
    all_labels = read_labels(label_file) if label_file else []
    label_to_id_fn = get_label_to_id_fn(all_labels)

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_random_seed = None if args.do_predict or args.do_export else random_seed
    with accelerator.main_process_first():
        train_dataloader, eval_dataloader, train_dataset, eval_dataset = get_dataloader(tokenizer, label_to_id_fn, dataset_random_seed, args)

    if args.do_export:
        model = EsmModel.from_pretrained(model_name)
    else:
        model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=len(all_labels))

    #num_samples_per_class = train_dataset.get_num_samples_per_class()
    #accelerator.print(f"num_samples_per_class: {num_samples_per_class}")

    loss_fn = get_focal_loss_fn(train_dataset, eval_dataset) if args.loss_fn == 'focal' else None

    with accelerator.accumulate(model):
        if args.do_train:
            process_training(model, tokenizer, train_dataloader, eval_dataloader, loss_fn, all_labels, accelerator, args)
        elif args.do_eval:
            model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
            do_eval(model, eval_dataloader, loss_fn, all_labels, accelerator)
        elif args.do_predict:
            model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
            do_predict(model, args.predict_head_mask, eval_dataloader, all_labels, args.output_file, accelerator)
        elif args.do_export:
            model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
            do_export(model, eval_dataloader, args.output_dir, accelerator, args.attention_strategy, args.attention_threshold, args.attention_layers)
        else:
            pass

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    print(args.attention_layers)
    main(args)

    
        