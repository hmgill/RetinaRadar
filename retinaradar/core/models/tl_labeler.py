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
            learning_rate=1e-3,
            label_names=None
    ):
        super().__init__()

        # This saves hyperparameters to self.hparams, and MLFlowLogger will autolog them
        self.save_hyperparameters()
        
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Store label names for organizing metrics
        self.label_names = label_names if label_names is not None else []
        
        # --- Initialize overall metrics ---
        self.train_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=0.5)
        self.train_f1_score = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=0.5)
        
        self.val_accuracy = torchmetrics.Accuracy(task="multilabel", num_labels=num_labels, threshold=0.5)
        self.val_f1_score = torchmetrics.F1Score(task="multilabel", num_labels=num_labels, threshold=0.5)
        
        # --- CRITICAL FIX: Initialize ModuleDicts BEFORE calling setup ---
        # Must use torch.nn.ModuleDict and initialize before setup
        self.category_metrics_train = torch.nn.ModuleDict()
        self.category_metrics_val = torch.nn.ModuleDict()
        
        # Initialize empty dict for category indices (not a module, just a regular dict)
        self.category_indices = {}
        
        # Then populate them
        self._setup_category_metrics()

    def _setup_category_metrics(self):
        """
        Set up per-category accuracy metrics based on the one-hot encoded label structure.
        Categories are: laterality, fundus_image_type, and quality fields (artifacts, clarity, etc.)
        """
        if not self.label_names:
            return
        
        # Map one-hot feature names to their logical categories
        # Based on the order in multilabel_array: [laterality, fundus_image_type, artifacts, clarity, illumination, contrast, field, usable]
        self.category_indices = {
            'laterality': [],
            'fundus_type': [],  # renamed from 'type' to avoid Python reserved keyword
            'artifacts': [],
            'clarity': [],
            'illumination': [],
            'contrast': [],
            'field': [],
            'usable': []
        }
        
        # Parse label names to determine which indices belong to which category
        # Label names from OneHotEncoder follow pattern: "x{feature_idx}_{value}"
        # where feature_idx maps to: 0=laterality, 1=fundus_type, 2=artifacts, 3=clarity, 4=illumination, 5=contrast, 6=field, 7=usable
        
        category_mapping = {
            'x0': 'laterality',
            'x1': 'fundus_type',
            'x2': 'artifacts',
            'x3': 'clarity',
            'x4': 'illumination',
            'x5': 'contrast',
            'x6': 'field',
            'x7': 'usable'
        }
        
        for idx, label_name in enumerate(self.label_names):
            # Extract the feature index (e.g., 'x0', 'x1', etc.)
            for prefix, category in category_mapping.items():
                if label_name.startswith(f'{prefix}_'):
                    self.category_indices[category].append(idx)
                    break
        
        # Populate the already-initialized ModuleDicts
        for category, indices in self.category_indices.items():
            if indices:  # Only create metric if category has labels
                num_category_labels = len(indices)
                
                # For single label categories, use binary classification
                # For multiple labels, use multilabel classification
                if num_category_labels == 1:
                    self.category_metrics_train[category] = torchmetrics.Accuracy(
                        task="binary",
                        threshold=0.5
                    )
                    self.category_metrics_val[category] = torchmetrics.Accuracy(
                        task="binary",
                        threshold=0.5
                    )
                else:
                    self.category_metrics_train[category] = torchmetrics.Accuracy(
                        task="multilabel", 
                        num_labels=num_category_labels, 
                        threshold=0.5
                    )
                    self.category_metrics_val[category] = torchmetrics.Accuracy(
                        task="multilabel", 
                        num_labels=num_category_labels, 
                        threshold=0.5
                    )

    def _compute_category_metrics(self, logits, labels, stage='train'):
        """
        Compute and log per-category accuracy metrics.
        
        Args:
            logits: Model output logits [batch_size, num_labels]
            labels: Ground truth labels [batch_size, num_labels]
            stage: 'train' or 'val'
        """
        category_metrics = self.category_metrics_train if stage == 'train' else self.category_metrics_val
        
        for category, indices in self.category_indices.items():
            if indices and category in category_metrics:
                # Extract only the logits and labels for this category
                category_logits = logits[:, indices]
                category_labels = labels[:, indices]
                
                # For single-label categories, squeeze to 1D for binary classification
                if len(indices) == 1:
                    category_logits = category_logits.squeeze(-1)
                    category_labels = category_labels.squeeze(-1)
                
                # Compute accuracy for this category
                acc = category_metrics[category](category_logits, category_labels)
                
                # Log with clean name
                self.log(f'{stage}_{category}_acc', acc, on_step=False, on_epoch=True, logger=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log training loss and overall metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy(logits, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', self.train_f1_score(logits, y), on_step=False, on_epoch=True, logger=True)
        
        # Log per-category metrics
        self._compute_category_metrics(logits, y, stage='train')
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Log validation loss and overall metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_accuracy(logits, y), on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.val_f1_score(logits, y), on_epoch=True, logger=True)
        
        # Log per-category metrics
        self._compute_category_metrics(logits, y, stage='val')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
