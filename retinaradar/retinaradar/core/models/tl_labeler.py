import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


class MultiLabelImageClassifier(pl.LightningModule):
    
    def __init__(
            self,
            model_name='resnet18',
            num_labels=10,
            learning_rate=1e-3
    ):
        super().__init__()

        # This saves hyperparameters to self.hparams, and MLFlowLogger will autolog them
        self.save_hyperparameters()
        
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # --- Initialize metrics using torchmetrics ---
        # We use a threshold of 0.5 to convert sigmoid outputs to binary predictions
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=0.5)
        self.f1_score = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=0.5)

        
    def forward(self, x):
        return self.model(x)

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log training loss and metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.accuracy(logits, y), on_step=False, on_epoch=True, logger=True)
        self.log('train_f1', self.f1_score(logits, y), on_step=False, on_epoch=True, logger=True)
        
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log validation loss and metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.accuracy(logits, y), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.f1_score(logits, y), on_epoch=True, logger=True)

        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
