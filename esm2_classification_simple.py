import argparse
from datetime import datetime
import json
import logging
import os
import gc

import yaml
from accelerate import Accelerator
from accelerate.utils import DummyScheduler, DummyOptim, set_seed
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanMetric
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForSequenceClassification
from transformers.trainer_pt_utils import get_parameter_names
import deepspeed
from deepspeed.accelerator import get_accelerator
import wandb
import shutil
from box import Box

from simple_utils.focalloss import FocalLoss
from simple_utils.data_utils import read_fasta, read_seq_labels_and_metadata, read_labels, get_sample_weights, get_label_convertion_fn

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
    parser.add_argument('-c', '--config', type=str, help='Path to the YAML config file')
    parser.add_argument('-a', '--action', type=str, required=True, help='action', choices=["train", "eval", "predict"])
    parser.add_argument('-o', '--output_dir', type=str, help='output directory')
    parser.add_argument('--output_file', type=str, help='predict/export output file')
    parser.add_argument('--output_loss_on_prediction', action='store_true', default=False, help='whether output loss on prediction')
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    return parser


class SequenceDataset(Dataset):
    def __init__(self, inputs, labels, names, metadata=None):
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.labels = torch.tensor(labels)
        self.names = names
        self.metadata = metadata
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        item = {'labels': self.labels[idx], 'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'ids': idx}
        if self.metadata:
            item['metadata'] = self.metadata[idx]
        return item
    def get_num_samples_per_class(self):
        if len(self.labels.shape) == 1:
            return torch.bincount(self.labels).tolist()
        else:
            return self.labels.sum(axis=0).tolist() # type: ignore

def create_dataset(tokenizer, fasta_dir, label_to_id_fn, random_seed, args):
    seq_labels, seq_metadata = read_seq_labels_and_metadata(args.seq_labels)
    labels, sequences, names, metadata = read_fasta(fasta_dir, seq_labels, seq_metadata)
    if random_seed is not None:
        if metadata:
            labels, sequences, names, metadata = shuffle(labels, sequences, names, metadata, random_state=random_seed) # type: ignore
        else:
            labels, sequences, names = shuffle(labels, sequences, names, random_state=random_seed) # type: ignore
    inputs = tokenizer(sequences, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors='pt', add_special_tokens=True)
    label_ids = label_to_id_fn(labels)
    return SequenceDataset(inputs, label_ids, names, metadata)

def get_dataloader_dataset(tokenizer, label_to_id_fn, random_seed, args):
    train_dataloader = None
    train_dataset = None
    validate_dataloader = None
    validate_dataset = None
    if args.do_train:
        train_dataset = create_dataset(tokenizer, args.train_dataset_dir, label_to_id_fn, random_seed, args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.micro_train_batch_size)
        validate_dataset = create_dataset(tokenizer, args.validate_dataset_dir, label_to_id_fn, random_seed, args)
        validate_dataloader = DataLoader(validate_dataset, batch_size=args.eval_batch_size)
    elif args.do_eval:
        validate_dataset = create_dataset(tokenizer, args.validate_dataset_dir, label_to_id_fn, random_seed, args)
        validate_dataloader = DataLoader(validate_dataset, batch_size=args.eval_batch_size)
    elif args.do_predict:
        pass
    else:
        raise ValueError("At least one of `do_train`, `do_eval`, `do_predict` must be True.")
    return train_dataloader, train_dataset, validate_dataloader, validate_dataset 

def get_optimizer(model, accelerator, args):
    fixed_param_layers = args.fixed_param_layers
    if fixed_param_layers:
        start = fixed_param_layers.start
        end = fixed_param_layers.end
        for param in model.esm.encoder.layer[start:end].parameters():
            param.requires_grad = False
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

def compute_metrics(labels, logits, logits_label_threshold):
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()}

def compute_class_weights(num_samples_per_class):
    epsilon=1e-6
    weights = [1 / (n + epsilon) for n in num_samples_per_class]
    normalized_weights = [w / sum(weights) for w in weights]
    return torch.tensor(normalized_weights)

def get_focal_loss_fn(train_dataset, validate_dataset):
    if train_dataset:
        num_samples_per_class = train_dataset.get_num_samples_per_class()
    else:
        num_samples_per_class = validate_dataset.get_num_samples_per_class()
    logger.info(f"num_samples_per_class: {num_samples_per_class}")
    alpha = compute_class_weights(num_samples_per_class)
    gamma = 2

    loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    return loss_fn

def get_loss_logits(outputs, labels, loss_fn):
    logits = outputs.get("logits")
    if loss_fn:
        loss = loss_fn(logits, labels)
    else:
        loss = outputs.get("loss")
    return loss, logits

def do_train(model, dataloader, loss_fn, optimizer, scheduler, accelerator, args, epoch):
    global wandb_run
    model.train()
    log_steps = args.logging_steps
    length = len(dataloader)
    inputs_keys = DATASET_TRAINING_KEYS.copy()
    if loss_fn:
        inputs_keys.remove("labels")
    for step, batch in enumerate(tqdm(dataloader), start=1):
        inputs = {k: v for k, v in batch.items() if k in inputs_keys}
        labels = batch.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if loss_fn:
            sample_weights = get_sample_weights(batch)
            loss = loss_fn(logits, labels, sample_weights)
        else:
            loss = outputs.get("loss")
        try:
            accelerator.backward(loss)
        except torch.cuda.OutOfMemoryError: # type: ignore
            # Occasionally, there may arise a dearth of memory.
            logger.warning("CUDA out of memory. Retrying the current batch.")
            get_accelerator().empty_cache()
        if step % accelerator.gradient_accumulation_steps == 0 or step == length:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        global_step = step + epoch * length
        lr = optimizer.param_groups[0]['lr']
        if global_step % log_steps == 0:
            accelerator.print(f"\nepoch: {epoch}: device: {accelerator.device}, step: {global_step}, loss: {loss}, lr: {lr}")
        if accelerator.is_main_process:
            if wandb_run:
                #wandb_run.log({ "train_loss": train_loss, "lr": lr, "global_step": global_step })
                wandb_run.log({ "train_loss": loss, "lr": lr, "global_step": global_step, "epoch": epoch, "device": f"{accelerator.device}" })
        if args.save_steps > 0 and global_step % args.save_steps == 0:
            accelerator.save_state(f"{args.output_dir}/step_{global_step}")
            previous_save_step = global_step - args.save_steps
            if previous_save_step > 0:
                accelerator.wait_for_everyone()
                accelerator.print(f"Deleting step: {previous_save_step}")
                shutil.rmtree(f"{args.output_dir}/step_{previous_save_step}", ignore_errors=True)


def do_eval(model, dataloader, loss_fn, accelerator, args, epoch=None):
    global wandb_run
    model.eval()
    merged_labels = torch.tensor([], dtype=torch.int)
    merged_logits = torch.tensor([], dtype=torch.float)
    mean_loss = MeanMetric(nan_strategy="error").to(accelerator.device)
    inputs_keys = DATASET_TRAINING_KEYS.copy()
    if loss_fn:
        inputs_keys.remove("labels")
    with torch.no_grad():
        eval_loss = 0
        for batch in tqdm(dataloader):
            inputs = {k: v for k, v in batch.items() if k in inputs_keys}
            labels = batch.get("labels")
            weights = get_sample_weights(batch)
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if loss_fn:
                loss = loss_fn(logits, labels, weights)
            else:
                loss = outputs.get("loss")
            gathered_loss = accelerator.gather_for_metrics((loss))
            mean_loss.update(gathered_loss)
            gathered_labels, gathered_logits = accelerator.gather((labels, logits))
            if accelerator.is_main_process:
                merged_labels = torch.cat((merged_labels, gathered_labels.cpu()), dim=0)
                merged_logits = torch.cat((merged_logits, gathered_logits.cpu()), dim=0)
    eval_loss = mean_loss.compute().item()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metrics = compute_metrics(merged_labels, merged_logits, args.logits_label_threshold)
        eval_info =f"eval_loss: {eval_loss} eval metrics: {metrics}"
        if epoch:
            eval_info = f"Epoch {epoch} - {eval_info}"
        accelerator.print(eval_info)
        if wandb_run:
            wandb_data = { "eval_loss": eval_loss, **metrics }
            if epoch:
                wandb_data["epoch"] = epoch
            wandb_run.log(wandb_data)

def do_predict(model, dataloader, all_labels, id_to_label_fn, accelerator, args, index=None):
    model.eval()
    merged_ids = torch.tensor([], dtype=torch.int)
    merged_label_ids = torch.tensor([], dtype=torch.int)
    merged_predicted_label_ids = torch.tensor([], dtype=torch.int)
    merged_logits = torch.tensor([], dtype=torch.float)
    inputs_keys = DATASET_TRAINING_KEYS.copy()
    # inputs_keys.remove("labels")
    output_loss_on_prediction = args.output_loss_on_prediction
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            inputs = {k: v for k, v in batch.items() if k in inputs_keys}
            label_ids = batch.get("labels")
            ids = batch.get("ids")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            predicted_probs = torch.softmax(logits, dim=1)
            predicted_label_ids = torch.argmax(logits, dim=1)
            gathered_ids, gathered_label_ids, gathered_predicted_label_ids, gathered_predicted_probs = accelerator.gather((ids, label_ids, predicted_label_ids, predicted_probs))
            if output_loss_on_prediction:
                loss = outputs.get("loss")
                accelerator.print(f"step: {step}, eval loss: {loss}")
            if accelerator.is_main_process:
                merged_ids = torch.cat((merged_ids, gathered_ids.cpu()), dim=0)
                merged_label_ids = torch.cat((merged_label_ids, gathered_label_ids.cpu()), dim=0)
                merged_predicted_label_ids = torch.cat((merged_predicted_label_ids, gathered_predicted_label_ids.cpu()), dim=0)
                merged_logits = torch.cat((merged_logits, gathered_predicted_probs.cpu()), dim=0)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged_labels = id_to_label_fn(merged_label_ids)
        merged_predicted_labels = id_to_label_fn(merged_predicted_label_ids)
        merged_names = [dataloader.dataset.names[id] for id in merged_ids]
        accelerator.print(f"merged_names: len: {len(merged_names)}, merged_labels: len: {len(merged_labels)}, merged_predicted_labels: len: {len(merged_predicted_labels)}, merged_logits: len: {len(merged_logits)}")
        df = pd.DataFrame({
            "name": merged_names,
            "label": merged_labels,
            "predicted_label": merged_predicted_labels,
        })
        for i, label in enumerate(all_labels):
            df[f"prob: {label}"] = [f"{round(prob[i].item() * 100, 2)}%" for prob in merged_logits]
        if index:
            base, ext = os.path.splitext(args.output_file)
            output_file = f"{base}.{index}{ext}"
        else:
            output_file = args.output_file
        df.to_csv(output_file, index=False)

def process_action(args):
    args.do_train = False
    args.do_eval = False
    args.do_predict = False
    if args.action == "train":
        assert args.train_dataset_dir is not None
        assert args.output_dir is not None
        assert args.label_file is not None
        assert args.output_file is None
        args.output_dir = f"{args.output_dir}/{args.model_name.split('/')[-1]}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        args.do_train = True
    elif args.action == "eval":
        assert args.validate_dataset_dir is not None
        assert args.label_file is not None
        assert args.output_file is None
        assert args.output_dir is None
        args.do_eval = True
    elif args.action == "predict":
        assert args.validate_dataset_dir is not None
        assert args.output_file is not None
        assert args.label_file is not None
        assert args.output_dir is None
        args.do_predict = True
    else:
        pass

def process_training(model, tokenizer, train_dataloader, validate_dataloader, loss_fn, accelerator, args):

    optimizer = get_optimizer(model, accelerator, args)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    scheduler = make_scheduler(args, accelerator, train_dataloader, optimizer)

    model, optimizer, train_dataloader, validate_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, validate_dataloader, scheduler
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
        wandb_run = wandb.init(config=args, project=args.wandb_project, resume="allow")

    for epoch in range(args.num_train_epochs):
        epoch_save_dir = f"{args.output_dir}/epoch_{epoch}"
        do_train(model, train_dataloader, loss_fn, optimizer, scheduler, accelerator, args, epoch)
        accelerator.wait_for_everyone()
        do_eval(model, validate_dataloader, loss_fn, accelerator, args, epoch)
        accelerator.wait_for_everyone()
        get_accelerator().empty_cache()
        gc.collect()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            epoch_save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(epoch_save_dir)
        gc.collect()
        accelerator.print(f"Saved model checkpoint to {args.output_dir}/epoch_{epoch}")
    with open(f"{args.output_dir}/config.yaml", "w") as f:
        f.write(args.to_yaml())
    with open(f"{args.output_dir}/deepspeed.json", "w") as f:
        json.dump(accelerator.state.deepspeed_plugin.deepspeed_config, f, indent=2)

def make_accelerator(args):
    accelerator = Accelerator()
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    if deepspeed_plugin:
        deepspeed.init_distributed()
        deepspeed_config = deepspeed_plugin.deepspeed_config # type: ignore
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
    args.all_labels = all_labels
    label_to_id_fn, id_to_label_fn = get_label_convertion_fn(all_labels)

    if args.do_predict:
        model_name = args.test.checkpoint
    else:
        model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_random_seed = None if args.do_predict else random_seed
    with accelerator.main_process_first():
        train_dataloader, train_dataset, validate_dataloader, validate_dataset = get_dataloader_dataset(tokenizer, label_to_id_fn, dataset_random_seed, args)

    model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=len(all_labels))

    if (args.do_train or args.do_eval) and args.loss_fn == 'focal':
        loss_fn = get_focal_loss_fn(train_dataset, validate_dataset)
    else:
        loss_fn = None

    if args.do_train:
        with accelerator.accumulate(model):
            process_training(model, tokenizer, train_dataloader, validate_dataloader, loss_fn, accelerator, args)
    elif args.do_eval:
        model, validate_dataloader = accelerator.prepare(model, validate_dataloader)
        do_eval(model, validate_dataloader, loss_fn, accelerator, args)
    elif args.do_predict:
        model = accelerator.prepare(model)
        if args.predict_data_file_start_index is not None and args.predict_num_data_file is not None and args.test.dataset_dirs_pattern is not None:
            for i in range(args.predict_num_data_file):
                index = args.predict_data_file_start_index + i
                accelerator.print(f"predict for index: {index}")
                dataset_dir = args.test.dataset_dirs_pattern.replace('{index}', str(index))
                dataset = create_dataset(tokenizer, dataset_dir, label_to_id_fn, random_seed, args)
                test_dataloader = accelerator.prepare(DataLoader(dataset, batch_size=args.eval_batch_size))
                do_predict(model, test_dataloader, all_labels, id_to_label_fn, accelerator, args, index)
        else:
            for index, dataset_dir in enumerate(args.test.dataset_dirs, start=1):
                index = args.test.start_index + index
                accelerator.print(f"predict for index: {index}")
                dataset = create_dataset(tokenizer, dataset_dir, label_to_id_fn, random_seed, args)
                test_dataloader = accelerator.prepare(DataLoader(dataset, batch_size=args.eval_batch_size))
                do_predict(model, test_dataloader, all_labels, id_to_label_fn, accelerator, args, index)
    else:
        pass


def build_args():
    parser = create_parser()
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    for key, value in vars(args).items():
        if value is not None or key not in config:
            config[key] = value
            print(f"Set {key} to {value}")
    return Box(config, default_box=True, default_box_attr=None)

if __name__ == '__main__':
    args = build_args()
    main(args)

