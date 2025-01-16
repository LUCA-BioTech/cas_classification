from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer, EsmModel, EsmForSequenceClassification
from transformers.models.esm.modeling_esm import EsmClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class AdvEsmForSequenceClassification(EsmForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.classifier = None
        self.simple_classifier = ClassificationHead(config, config.hidden_size)
        self.concat_classifier = ClassificationHead(config, config.hidden_size*2)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        domain_classification_strategy: Optional[str] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        classifier, features = self.get_classifier_and_features(sequence_output, start_positions, end_positions, domain_classification_strategy)
        logits = classifier(features)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device) # type: ignore

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int): # type: ignore
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze()) # type: ignore
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) # type: ignore
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_classifier_and_features(self, features, start_positions, end_positions, domain_classification_strategy):
        classifier = None
        if start_positions is None or end_positions is None:
            selected_features = features
            classifier = self.simple_classifier
        else:
            selected_features = []
            if domain_classification_strategy == 'domain_mean':
                classifier = self.simple_classifier
                for i in range(features.size(0)):
                    domain_mean = features[i, start_positions[i]:end_positions[i]].mean(dim=0)
                    selected_features.append(domain_mean.unsqueeze(0))
            elif domain_classification_strategy == 'domain_mean_and_seq_cls':
                classifier = self.concat_classifier
                for i in range(features.size(0)):
                    domain_mean = features[i, start_positions[i]:end_positions[i]].mean(dim=0)
                    new_cls = torch.cat((domain_mean, features[i, 0]), dim=0)
                    selected_features.append(new_cls.unsqueeze(0))
            selected_features = torch.stack(selected_features)
        if classifier:
            return classifier, selected_features
        else:
            raise ValueError(f"domain_classification_strategy must be one of domain_mean, domain_mean_and_seq_cls, geot {domain_classification_strategy}")


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

