# bert_model.py

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAccuracy
from transformers import AutoConfig, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class LMModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path,
        vocab_size,
        pad_token_id=3,
        learning_rate=1e-5,
        lr_monitor_metric='valid_loss',
        LABEL_SIZE=12,
        pos_weight=None,
        class_weight=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(model_name_or_path)
        config.vocab_size = vocab_size
        config.pad_token_id = pad_token_id

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True
        )

        self.LABEL_SIZE = LABEL_SIZE
        self.dense_layer = nn.Linear(config.hidden_size, 1)
        self.self_ffn = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffn = nn.Linear(config.hidden_size, LABEL_SIZE)

        self.learning_rate = learning_rate
        self.lr_monitor_metric = lr_monitor_metric

        if pos_weight is None:
            self.pos_weight = torch.ones(LABEL_SIZE)
        else:
            self.pos_weight = pos_weight

        if class_weight is None:
            self.class_weight = torch.ones(LABEL_SIZE)
        else:
            self.class_weight = class_weight

        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.multilabel_acc = MultilabelAccuracy(num_labels=LABEL_SIZE, average=None)

    def forward(self, x):
        # x is a dict with 'input_ids' and 'attention_mask'
        bert_out = self.model(
            x['input_ids'], attention_mask=x['attention_mask']
        ).last_hidden_state

        # Attention mechanism
        attention_vector = self.dense_layer(
            bert_out.view(-1, bert_out.size(-1))
        )  # shape: [batch_size*seq_len, 1]
        attention_vector = attention_vector.view(
            bert_out.size(0), -1, 1
        )  # shape: [batch_size, seq_len, 1]
        attention_vector = attention_vector.squeeze(-1)  # shape: [batch_size, seq_len]
        attention_vector = F.softmax(attention_vector, dim=1)

        # Weighted sum of hidden states
        # bert_out: [batch_size, seq_len, hidden_dim]
        # attention_vector: [batch_size, seq_len]
        # output shape: [batch_size, hidden_dim]
        attention_output = torch.matmul(
            bert_out.transpose(-2, -1),  # [batch_size, hidden_dim, seq_len]
            attention_vector.unsqueeze(-1),
        ).squeeze(-1)

        fc = F.relu(self.self_ffn(attention_output))
        out = self.ffn(fc)  # shape: [batch_size, LABEL_SIZE]
        return torch.sigmoid(out)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss(logits, batch['label'])
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss(logits, batch['label'])

        self.log('valid_loss', loss, on_step=True, prog_bar=True)
        binary_logits = (logits >= 0.5).float().detach().cpu()
        labels = batch['label'].detach().cpu()
        self.validation_step_outputs.append({'logit': binary_logits, 'labels': labels})
        return loss

    def on_validation_epoch_end(self):
        logits = torch.tensor([])
        targets = torch.tensor([])

        for output in self.validation_step_outputs:
            logits = torch.cat((logits, output['logit']), dim=0)
            targets = torch.cat((targets, output['labels']), dim=0)

        precision = precision_score(targets, logits, average='macro', zero_division=1)
        recall = recall_score(targets, logits, average='macro', zero_division=1)
        f1 = f1_score(targets, logits, average='macro', zero_division=1)
        acc = accuracy_score(targets, logits)

        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': self.learning_rate},
            {'params': self.ffn.parameters(), 'lr': self.learning_rate},
        ])
        scheduler = ReduceLROnPlateau(optimizer, patience=2)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.lr_monitor_metric}
