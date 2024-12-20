"""
CholecT50 Surgical Workflow Recognition Model.

A PyTorch Lightning implementation of a multi-task learning model for surgical workflow recognition
in laparoscopic surgery videos. The model performs simultaneous detection of:
- Surgical instruments (tools)
- Actions (verbs)
- Instrument-verb pairs (IV)
- Surgical phases

The architecture uses a ResNet50 backbone with task-specific branches and implements an 
instrument-guided attention mechanism for improved performance.

Example:
    >>> from surgical_workflow.models import CholecT50Model
    >>> model = CholecT50Model()
    >>> trainer = pl.Trainer(max_epochs=100)
    >>> trainer.fit(model, train_dataloader, val_dataloader)

Attributes:
    NUM_TOOLS (int): Number of surgical instrument classes (default: 6)
    NUM_VERBS (int): Number of action classes (default: 10)
    NUM_IV (int): Number of instrument-verb pair classes (default: 26)
    NUM_PHASES (int): Number of surgical phase classes (default: 7)

References:
    [1]: C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. Medical Image Analysis, 78 (2022) 102433.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50
from sklearn.metrics import average_precision_score
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for the CholecT50 model.
    
    This class centralizes all configuration parameters for the model architecture
    and training process.
    
    Attributes:
        num_tools: Number of surgical instrument classes
        num_verbs: Number of action classes
        num_iv: Number of instrument-verb pair classes
        num_phases: Number of surgical phase classes
        feature_dim: Dimension of backbone features
        hidden_dim: Dimension of hidden layers
        dropout_rate: Dropout probability
    """
    # Model architecture
    num_tools: int = 6
    num_verbs: int = 10
    num_iv: int = 26
    num_phases: int = 7
    feature_dim: int = 2048
    hidden_dim: int = 1024
    dropout_rate: float = 0.5
    
    # Learning rates
    learning_rates: Dict[str, float] = field(default_factory=lambda: {
        'backbone': 1e-3,
        'tool': 5e-5,
        'verb': 1e-3,
        'iv': 3e-4,
        'phase': 5e-4
    })
    
    # Task weights for loss computation
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'tool': 0.8,
        'verb': 1.3,
        'iv': 1.7,
        'phase': 1.0
    })


def get_class_weights() -> Dict[str, torch.Tensor]:
    """Get class weights for handling imbalanced data.
    
    Returns:
        Dictionary containing weight tensors for tool and verb classes
    """
    weights = {
        'tool': torch.tensor([
            0.08495163,  # Grasper
            0.88782288,  # Bipolar
            0.11259564,  # Hook
            2.61948830,  # Scissors
            1.78486647,  # Clipper
            1.14462417   # Irrigator
        ]),
        'verb': torch.tensor([
            0.39862805,  # grasp
            0.06981640,  # retract
            0.08332925,  # dissect
            0.81876204,  # coagulate
            1.41586839,  # clip
            2.26935915,  # cut
            1.28428410,  # aspirate
            7.35822511,  # irrigate
            18.6785714,  # pack
            0.45704490   # null_verb
        ])
    }
    return weights

class InstrumentGuidedAttention(nn.Module):
    """Attention mechanism guided by instrument predictions.
    
    This module implements a spatial attention mechanism that uses instrument
    detection results to guide the network's focus on relevant image regions.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        """Initialize attention module.
        
        Args:
            feature_dim: Number of input feature channels
            hidden_dim: Number of hidden channels in attention computation
        """
        super().__init__()
        
        self.attention_net = nn.Sequential(
            # Reduce feature dimensions
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Compute attention weights
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, tool_features: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism.
        
        Args:
            features: Input feature maps [batch_size, channels, height, width]
            tool_features: Tool detection features [batch_size, num_tools, height, width]
            
        Returns:
            Attended feature maps with same shape as input features
        """
        # Concatenate feature maps
        combined_features = torch.cat([features, tool_features], dim=1)
        
        # Compute attention weights
        attention_weights = self.attention_net(combined_features)
        
        # Apply attention
        attended_features = features * attention_weights
        
        return attended_features


class TaskBranch(nn.Module):
    """Generic branch for task-specific predictions.
    
    This module implements a reusable architecture for different prediction tasks
    (tools, verbs, IV pairs, phases) with optional guidance from other tasks.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guidance_channels: Optional[int] = None,
        hidden_channels: int = 512,
        dropout_rate: float = 0.5
    ):
        """Initialize task branch.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output classes
            guidance_channels: Number of guidance feature channels (optional)
            hidden_channels: Number of hidden channels
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Calculate total input channels
        total_channels = in_channels
        if guidance_channels is not None:
            total_channels += guidance_channels
        
        # Main prediction network
        self.network = nn.Sequential(
            # First convolution block
            nn.Conv2d(total_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            
            # Second convolution block
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            
            # Final prediction layer
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        guidance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through task branch.
        
        Args:
            x: Input features [batch_size, in_channels, height, width]
            guidance: Optional guidance features [batch_size, guidance_channels, height, width]
            
        Returns:
            Task predictions [batch_size, out_channels]
        """
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        
        return self.network(x)


class InstrumentVerbMapping(nn.Module):
    """Module for instrument-verb compatibility mapping.
    
    This module maintains and applies the compatibility matrix between
    instruments and verbs to ensure valid instrument-verb combinations.
    """
    
    def __init__(self):
        """Initialize the instrument-verb mapping matrix."""
        super().__init__()
        
        # Define the compatibility matrix
        mapping = torch.tensor([
            # Grasper
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            # Bipolar
            [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            # Hook
            [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            # Scissors
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            # Clipper
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            # Irrigator
            [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]
        ], dtype=torch.float32)
        
        # Register as buffer to save with model
        self.register_buffer('mapping', mapping)
    
    def forward(self, tool_probabilities: torch.Tensor) -> torch.Tensor:
        """Apply instrument-verb compatibility mapping.
        
        Args:
            tool_probabilities: Tool detection probabilities [batch_size, num_tools]
            
        Returns:
            Verb compatibility mask [batch_size, num_verbs]
        """
        return torch.mm(tool_probabilities, self.mapping)
    
class CholecT50Model(pl.LightningModule):
        """
        Main model for surgical workflow recognition.
        This PyTorch Lightning module combines all components for multi-task learning:
        - Feature extraction (ResNet50 backbone)
        - Instrument-guided attention
        - Task-specific prediction branches
        - Training and validation logic
        """
        
        def __init__(self, config: Optional[ModelConfig] = None):
            """Initialize the model.
            
            Args:
                config: Model configuration, uses default if not provided
            """
            super().__init__()
            self.config = config or ModelConfig()
            self.save_hyperparameters()
            
            # Build model components
            self._build_backbone()
            self._build_task_branches()
            self._init_criterions()
            self._init_metric_tracking()

        def _build_backbone(self):
            """Initialize feature extraction backbone."""
            # Load pretrained ResNet50
            resnet = resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            
            # Freeze early layers
            for param in self.backbone[:-3].parameters():
                param.requires_grad = False
                
            # Feature refinement
            self.feature_refinement = nn.Sequential(
                nn.Conv2d(self.config.feature_dim, self.config.hidden_dim, 1),
                nn.BatchNorm2d(self.config.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.config.dropout_rate)
            )
            
            # Attention mechanism
            self.attention = InstrumentGuidedAttention(
                feature_dim=self.config.hidden_dim + self.config.num_tools
            )

        def _build_task_branches(self):
            """Initialize task-specific branches."""
            # Tool detection branch
            self.tool_branch = TaskBranch(
                in_channels=self.config.hidden_dim,
                out_channels=self.config.num_tools
            )
            
            # Verb recognition branch (guided by tools)
            self.verb_branch = TaskBranch(
                in_channels=self.config.hidden_dim,
                out_channels=self.config.num_verbs,
                guidance_channels=self.config.num_tools
            )
            
            # IV recognition branch (guided by tools and verbs)
            self.iv_branch = TaskBranch(
                in_channels=self.config.hidden_dim,
                out_channels=self.config.num_iv,
                guidance_channels=self.config.num_tools + self.config.num_verbs
            )
            
            # Phase recognition branch (guided by tools)
            self.phase_branch = TaskBranch(
                in_channels=self.config.hidden_dim,
                out_channels=self.config.num_phases,
                guidance_channels=self.config.num_tools
            )
            
            # Instrument-verb mapping
            self.iv_mapping = InstrumentVerbMapping()

        def _init_criterions(self):
            """Initialize loss functions."""
            weights = get_class_weights()
            self.criterion_tool = nn.BCEWithLogitsLoss(pos_weight=weights['tool'])
            self.criterion_verb = nn.BCEWithLogitsLoss(pos_weight=weights['verb'])
            self.criterion_iv = nn.BCEWithLogitsLoss()
            self.criterion_phase = nn.BCEWithLogitsLoss()

        def _init_metric_tracking(self):
            """Initialize metrics for tracking."""
            self.predictions = {
                "iv": [], "tools": [], "verbs": [], "phases": []
            }
            self.targets = {
                "iv": [], "tools": [], "verbs": [], "phases": []
            }

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            """Forward pass through the model.
            
            Args:
                x: Input images [batch_size, channels, height, width]
                
            Returns:
                Tuple of (iv_preds, tool_preds, verb_preds, phase_preds)
            """
            # Extract features
            features = self.backbone(x)
            features = self.feature_refinement(features)
            
            # Tool prediction
            tool_preds = self.tool_branch(features)
            tool_probs = torch.sigmoid(tool_preds)
            
            # Apply attention with tool guidance
            tool_attention = tool_probs.unsqueeze(-1).unsqueeze(-1)
            attended_features = self.attention(features, tool_attention.expand(-1, -1, *features.shape[-2:]))
            
            # Verb prediction with tool guidance
            verb_preds = self.verb_branch(attended_features, tool_attention)
            
            # Apply instrument-verb compatibility
            verb_probs = torch.sigmoid(verb_preds)
            verb_mask = self.iv_mapping(tool_probs)
            masked_verb_probs = verb_probs * verb_mask
            
            # Combine guidance for IV prediction
            combined_guidance = torch.cat([
                tool_attention,
                masked_verb_probs.unsqueeze(-1).unsqueeze(-1)
            ], dim=1)
            
            # IV and phase predictions
            iv_preds = self.iv_branch(attended_features, combined_guidance)
            phase_preds = self.phase_branch(attended_features, tool_attention)
            
            return iv_preds, tool_preds, verb_preds, phase_preds

        def configure_optimizers(self):
            """Configure optimizer and learning rate scheduler."""
            # Group parameters by learning rate
            param_groups = [
                {'params': self.backbone.parameters(), 
                'lr': self.config.learning_rates['backbone']},
                {'params': self.tool_branch.parameters(), 
                'lr': self.config.learning_rates['tool']},
                {'params': self.verb_branch.parameters(), 
                'lr': self.config.learning_rates['verb']},
                {'params': self.iv_branch.parameters(), 
                'lr': self.config.learning_rates['iv']},
                {'params': self.phase_branch.parameters(), 
                'lr': self.config.learning_rates['phase']}
            ]
            
            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                verbose=True
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        def compute_losses(
            self,
            predictions: Tuple[torch.Tensor, ...],
            targets: Tuple[torch.Tensor, ...]
        ) -> Dict[str, torch.Tensor]:
            """Compute task-specific and combined losses.
            
            Args:
                predictions: Model predictions (iv, tool, verb, phase)
                targets: Ground truth labels (iv, tool, verb, phase)
                
            Returns:
                Dictionary containing individual and total losses
            """
            iv_preds, tool_preds, verb_preds, phase_preds = predictions
            iv_target, tool_target, verb_target, phase_target = targets
            
            # Compute individual losses
            losses = {
                'tool': self.criterion_tool(tool_preds, tool_target),
                'verb': self.criterion_verb(verb_preds, verb_target),
                'iv': self.criterion_iv(iv_preds, iv_target),
                'phase': self.criterion_phase(phase_preds, phase_target)
            }
            
            # Compute weighted total loss
            total_loss = sum(
                self.config.task_weights[task] * loss 
                for task, loss in losses.items()
            )
            
            losses['total'] = total_loss
            return losses

        def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
            """Execute training step.
            
            Args:
                batch: Tuple of (images, labels)
                batch_idx: Index of current batch
                
            Returns:
                Total loss value
            """
            # Unpack batch
            images, (iv_target, tool_target, verb_target, _, phase_target) = batch
            
            # Forward pass
            predictions = self(images)
            
            # Compute losses
            losses = self.compute_losses(
                predictions,
                (iv_target, tool_target, verb_target, phase_target)
            )
            
            # Log training metrics
            self.log('train/loss', losses['total'])
            for task, loss in losses.items():
                if task != 'total':
                    self.log(f'train/{task}_loss', loss)
            
            return losses['total']

        def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> None:
            """Execute validation step.
            
            Args:
                batch: Tuple of (images, labels)
                batch_idx: Index of current batch
            """
            # Unpack batch
            images, (iv_target, tool_target, verb_target, _, phase_target) = batch
            
            # Forward pass
            predictions = self(images)
            iv_preds, tool_preds, verb_preds, phase_preds = predictions
            
            # Compute losses
            losses = self.compute_losses(
                predictions,
                (iv_target, tool_target, verb_target, phase_target)
            )
            
            # Log validation metrics
            self.log('val/loss', losses['total'])
            for task, loss in losses.items():
                if task != 'total':
                    self.log(f'val/{task}_loss', loss)
            
            # Store predictions and targets for mAP computation
            self._store_predictions(
                predictions=(iv_preds, tool_preds, verb_preds, phase_preds),
                targets=(iv_target, tool_target, verb_target, phase_target)
            )

        def _store_predictions(
            self,
            predictions: Tuple[torch.Tensor, ...],
            targets: Tuple[torch.Tensor, ...]
        ) -> None:
            """Store predictions and targets for metric computation.
            
            Args:
                predictions: Model predictions
                targets: Ground truth labels
            """
            # Map predictions and targets to their respective tasks
            pred_dict = {
                'iv': predictions[0],
                'tools': predictions[1],
                'verbs': predictions[2],
                'phases': predictions[3]
            }
            target_dict = {
                'iv': targets[0],
                'tools': targets[1],
                'verbs': targets[2],
                'phases': targets[3]
            }
            
            # Store as numpy arrays
            for task in self.predictions.keys():
                self.predictions[task].append(
                    torch.sigmoid(pred_dict[task]).detach().cpu().numpy()
                )
                self.targets[task].append(
                    target_dict[task].detach().cpu().numpy()
                )

        def on_validation_epoch_end(self) -> None:
            """Compute metrics at the end of validation epoch."""
            metrics = self._compute_metrics()
            
            # Log metrics
            for name, value in metrics.items():
                self.log(f'val/{name}', value)
            
            # Reset storage
            self._init_metric_tracking()

        def _compute_metrics(self) -> Dict[str, float]:
            """Compute evaluation metrics (mAP).
            
            Returns:
                Dictionary containing computed metrics
            """
            metrics = {}
            
            # Compute mAP for each task
            for task in self.predictions.keys():
                # Concatenate predictions and targets
                y_pred = np.concatenate(self.predictions[task])
                y_true = np.concatenate(self.targets[task])
                
                # Compute AP for each class
                with np.errstate(invalid='ignore'):
                    ap_scores = average_precision_score(
                        y_true, y_pred, average=None
                    )
                    
                    # Handle NaN scores (classes not present in validation set)
                    ap_scores = np.nan_to_num(ap_scores, 0)
                    
                    # Compute mean AP
                    map_score = np.mean(ap_scores)
                
                # Store metrics
                metrics[f'mAP_{task}'] = float(map_score)
                for i, ap in enumerate(ap_scores):
                    metrics[f'AP_{task}_{i}'] = float(ap)
            
            return metrics

        def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> None:
            """Execute test step (same as validation)."""
            return self.validation_step(batch, batch_idx)

        def on_test_epoch_end(self) -> None:
            """Compute metrics at the end of test epoch."""
            return self.on_validation_epoch_end()