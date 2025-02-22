import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from sklearn.metrics import average_precision_score
import numpy as np
import warnings
from .archs.backbones import get_backbone

def get_weight_balancing(case='cholect50'):
    """
    Returns class weight balancing factors for different surgical action recognition tasks.
    
    Args:
        case (str): Dataset identifier. Currently supports 'cholect50-challenge'.
        
    Returns:
        dict: Dictionary containing weight balancing factors for tools and verbs.
    """
    switcher = {
        'cholect50-challenge': {
            'tool': [0.08495163, 0.88782288, 0.11259564, 2.61948830, 1.784866470, 1.144624170],
            'verb': [0.39862805, 0.06981640, 0.08332925, 0.81876204, 1.415868390, 2.269359150, 
                    1.28428410, 7.35822511, 18.67857143, 0.45704490],
            'iv':[1.29, 5.18, 22.24, 0.1, 1.36, 4.1, 47.77, 23.31, 9.02, 42.06, 0.12, 8.34, 227.62, 3.51, 25.8, 1.68, 1.81, 14.38, 9.37, 10.57, 0.51, 10.43, 1.54, 36.5, 24.34, 10.18],
            'phase': [19.16, 2.44, 11.84, 3.57, 22.23, 13.73, 17.72],
        },
    }
    return switcher.get(case)

class InstrumentGuidedMultiTaskModel(nn.Module):
    """
    Multi-task learning model for surgical workflow recognition with instrument guidance.
    
    The model uses a flexible backbone architecture and performs four related tasks:
    1. Tool detection
    2. Verb (action) recognition
    3. Instrument-Verb (IV) pair recognition
    4. Phase recognition
    
    The architecture implements a hierarchical structure where tool predictions guide
    the verb and phase recognition, while both tool and verb predictions guide the IV recognition.
    """
    
    def __init__(self, num_tools, num_verbs, num_iv, num_phases, backbone_name='resnet50', pretrained=True):
        """
        Initialize the multi-task model.
        
        Args:
            num_tools (int): Number of surgical tool classes
            num_verbs (int): Number of action/verb classes
            num_iv (int): Number of instrument-verb pair classes
            num_phases (int): Number of surgical phase classes
            backbone_name (str): Name of the backbone architecture to use
            pretrained (bool): Whether to use pretrained weights for the backbone
        """
        super().__init__()
        
        # Feature Extractor (Flexible Backbone)
        self.feature_extractor = get_backbone(backbone_name, pretrained=pretrained)
        backbone_channels = self.feature_extractor.get_output_channels()
        
        # Shared Feature Processing Layers
        self.shared_layers = nn.Sequential(
            nn.Conv2d(backbone_channels, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )
        
        # Tool Recognition Branch
        self.tool_branch = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.6),
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, num_tools, kernel_size=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        
        # Verb Recognition Branch (guided by Tool activations)
        self.verb_branch = nn.Sequential(
            nn.Conv2d(1024 + num_tools, 512, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.6),
            nn.Conv2d(512, num_verbs, kernel_size=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        
        # IV Recognition Branch (guided by Tool and Verb activations)
        self.iv_branch = nn.Sequential(
            nn.Conv2d(1024 + num_tools + num_verbs, 512, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.6),
            nn.Conv2d(512, num_iv, kernel_size=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        
        # Phase Recognition Branch (guided by Tool activations)
        self.phase_branch = nn.Sequential(
            nn.Conv2d(1024 + num_tools, 512, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.6),
            nn.Conv2d(512, num_phases, kernel_size=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        # Define valid Instrument-Verb combinations
        self.register_buffer('instrument_verb_mapping', torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # Grasper
            [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],  # Bipolar
            [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],  # Hook
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 0],  # Scissors
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Clipper
            [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],  # Irrigator
        ], dtype=torch.float32))
        
        # Initialize weights for the new layers
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for all layers except the backbone"""
        for m in [self.shared_layers, self.tool_branch, self.verb_branch, 
                 self.iv_branch, self.phase_branch]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def freeze_backbone(self):
        """Freeze backbone layers"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone layers"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            tuple: Predictions for IV pairs, tools, verbs, and phases
        """
        # Extract and process shared features
        features = self.feature_extractor(x)
        shared = self.shared_layers(features)
        
        # Tool recognition
        tool_output = self.tool_branch(shared)
        tool_activations = torch.sigmoid(tool_output).unsqueeze(-1).unsqueeze(-1)
        
        # Create tool-guided features
        tool_guided_features = torch.cat(
            [shared, tool_activations.expand(-1, -1, shared.size(2), shared.size(3))], 
            dim=1
        )
        
        # Verb recognition with tool guidance
        verb_output = self.verb_branch(tool_guided_features)
        verb_activations = torch.sigmoid(verb_output).unsqueeze(-1).unsqueeze(-1)
        
        # Apply instrument-verb compatibility mask
        verb_mask = torch.mm(torch.sigmoid(tool_output), self.instrument_verb_mapping)
        masked_verb_output = verb_output * verb_mask
        
        # IV recognition with tool and verb guidance
        iv_guided_features = torch.cat(
            [tool_guided_features, verb_activations.expand(-1, -1, shared.size(2), shared.size(3))], 
            dim=1
        )
        iv_output = self.iv_branch(iv_guided_features)
        
        # Phase recognition with tool guidance
        phase_output = self.phase_branch(tool_guided_features)
        
        return iv_output, tool_output, masked_verb_output, phase_output

class CholecT50Model(pl.LightningModule):
    """
    PyTorch Lightning module for training and evaluating the CholecT50 model.
    
    This class handles the training loop, loss computation, optimization,
    and evaluation metrics for the multi-task surgical workflow recognition model.
    """
    
    def __init__(self, learning_rate=1e-3, lr_tool=5e-5, lr_verb=1e-3, lr_iv=3e-4, lr_phase=5e-4):
        """
        Initialize the Lightning module.
        
        Args:
            learning_rate (float): Base learning rate for shared layers
            lr_tool (float): Learning rate for tool recognition branch
            lr_verb (float): Learning rate for verb recognition branch
            lr_iv (float): Learning rate for IV pair recognition branch
            lr_phase (float): Learning rate for phase recognition branch
        """
        super().__init__()
        
        # Define model dimensions
        num_tools = 6
        num_verbs = 10
        num_iv = 26
        num_phases = 7
        
        self.model = InstrumentGuidedMultiTaskModel(num_tools, num_verbs, num_iv, num_phases)
        
        # Load class weight balancing factors
        weights = get_weight_balancing('cholect50-challenge')
        self.tool_weights = torch.tensor(weights['tool'])
        self.verb_weights = torch.tensor(weights['verb'])
        self.iv_weights = torch.tensor(weights['iv'])
        self.phase_weights = torch.tensor(weights['phase'])  
        
        # Define loss functions
        self.tool_criterion = nn.BCEWithLogitsLoss(pos_weight=self.tool_weights)
        self.verb_criterion = nn.BCEWithLogitsLoss(pos_weight=self.verb_weights)
        self.iv_criterion = nn.BCEWithLogitsLoss(pos_weight=self.iv_weights)
        self.phase_criterion = nn.BCEWithLogitsLoss(pos_weight=self.phase_weights)
        
        # Store learning rates
        self.learning_rate = learning_rate
        self.lr_tool = lr_tool
        self.lr_verb = lr_verb
        self.lr_iv = lr_iv
        self.lr_phase = lr_phase
        
        # Initialize prediction and target collectors for metric computation
        self.predictions = {"iv": [], "tools": [], "verbs": [], "phases": []}
        self.targets = {"iv": [], "tools": [], "verbs": [], "phases": []}

        # Initial calculation of the task weights
        self.calculate_task_weights()

    def calculate_task_weights(self):
        # Calculate variance/spread of weights for each task
        iv_spread = (self.iv_weights.max() - self.iv_weights.min()) / self.iv_weights.mean()
        tool_spread = (self.tool_weights.max() - self.tool_weights.min()) / self.tool_weights.mean()
        verb_spread = (self.verb_weights.max() - self.verb_weights.min()) / self.verb_weights.mean()
        phase_spread = (self.phase_weights.max() - self.phase_weights.min()) / self.phase_weights.mean()
        
        # Calculate task weights based on the variance
        total_spread = iv_spread + tool_spread + verb_spread + phase_spread
        self.iv_weight = iv_spread / total_spread
        self.tool_weight = tool_spread / total_spread
        self.verb_weight = verb_spread / total_spread
        self.phase_weight = phase_spread / total_spread

    def normalize_losses(self, iv_loss, tool_loss, verb_loss, phase_loss):
        """Normalisiert die Losses basierend auf den Klassengewichten"""
        # Normalisiere jeden Loss durch die durchschnittlichen Gewichte
        norm_iv_loss = iv_loss / self.iv_weights.mean()
        norm_tool_loss = tool_loss / self.tool_weights.mean()
        norm_verb_loss = verb_loss / self.verb_weights.mean()
        norm_phase_loss = phase_loss / self.phase_weights.mean()
        return norm_iv_loss, norm_tool_loss, norm_verb_loss, norm_phase_loss    

    def forward(self, x):
        """Forward pass wrapper for the underlying model."""
        return self.model(x)
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            dict: Optimizer and learning rate scheduler configuration
        """
        # Define parameter groups with different learning rates
        params = [
            {'params': self.model.feature_extractor.parameters(), 'lr': self.learning_rate},
            {'params': self.model.shared_layers.parameters(), 'lr': self.learning_rate},
            {'params': self.model.tool_branch.parameters(), 'lr': self.lr_tool},
            {'params': self.model.verb_branch.parameters(), 'lr': self.lr_verb},
            {'params': self.model.iv_branch.parameters(), 'lr': self.lr_iv},
            {'params': self.model.phase_branch.parameters(), 'lr': self.lr_phase},
        ]
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(params, weight_decay=1e-2)
        
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=2, factor=0.5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
            },
        }
    
    def training_step(self, batch, batch_idx):
        """
        Training step logic.
        
        Args:
            batch (tuple): Input batch containing images and labels
            batch_idx (int): Index of the current batch
            
        Returns:
            dict: Dictionary containing loss values
        """
        img, labels = batch
        iv_label, tool_label, verb_label, _, phase_label = labels
        
        # Forward pass
        iv_output, tool_output, verb_output, phase_output = self(img)
        
        # Compute individual losses
        iv_loss = self.iv_criterion(iv_output, iv_label.float())
        tool_loss = self.tool_criterion(tool_output, tool_label.float())
        verb_loss = self.verb_criterion(verb_output, verb_label.float())
        phase_loss = self.phase_criterion(phase_output, phase_label.float())
        
        # Normalize losses based on class weights
        norm_iv_loss, norm_tool_loss, norm_verb_loss, norm_phase_loss = self.normalize_losses(
            iv_loss, tool_loss, verb_loss, phase_loss
        )
        
        # Compute weighted total loss using dynamic task weights
        total_loss = (
            self.iv_weight * norm_iv_loss +
            self.tool_weight * norm_tool_loss +
            self.verb_weight * norm_verb_loss +
            self.phase_weight * norm_phase_loss
        )
        
        # Log individual losses
        self.log("train/iv_loss", iv_loss)
        self.log("train/tool_loss", tool_loss)
        self.log("train/verb_loss", verb_loss)
        self.log("train/phase_loss", phase_loss)
        self.log("train/total_loss", total_loss)
        
        # Log task weights
        self.log("train/weight_iv", self.iv_weight)
        self.log("train/weight_tool", self.tool_weight)
        self.log("train/weight_verb", self.verb_weight)
        self.log("train/weight_phase", self.phase_weight)
        
        # Log normalized losses
        self.log("train/norm_iv_loss", norm_iv_loss)
        self.log("train/norm_tool_loss", norm_tool_loss)
        self.log("train/norm_verb_loss", norm_verb_loss)
        self.log("train/norm_phase_loss", norm_phase_loss)
        
        return {
            "loss": total_loss,
            "iv_loss": iv_loss,
            "tool_loss": tool_loss,
            "verb_loss": verb_loss,
            "phase_loss": phase_loss,
            "norm_iv_loss": norm_iv_loss,
            "norm_tool_loss": norm_tool_loss,
            "norm_verb_loss": norm_verb_loss,
            "norm_phase_loss": norm_phase_loss
        }
    def validation_step(self, batch, batch_idx):
        """
        Validation step logic.
        
        Similar to training step but also accumulates predictions for metric computation.
        """
        img, labels = batch
        iv_label, tool_label, verb_label, _, phase_label = labels
        
        # Forward pass
        iv_output, tool_output, verb_output, phase_output = self(img)
        
        # Compute losses
        iv_loss = self.iv_criterion(iv_output, iv_label.float())
        tool_loss = self.tool_criterion(tool_output, tool_label.float())
        verb_loss = self.verb_criterion(verb_output, verb_label.float())
        phase_loss = self.phase_criterion(phase_output, phase_label.float())
        
        # Normalize losses based on class weights
        norm_iv_loss, norm_tool_loss, norm_verb_loss, norm_phase_loss = self.normalize_losses(
            iv_loss, tool_loss, verb_loss, phase_loss
        )
        
        # Compute weighted total loss using dynamic task weights
        total_loss = (
            self.iv_weight * norm_iv_loss +
            self.tool_weight * norm_tool_loss +
            self.verb_weight * norm_verb_loss +
            self.phase_weight * norm_phase_loss
        )
        
        # Log validation losses
        self.log("val/iv_loss", iv_loss)
        self.log("val/tool_loss", tool_loss)
        self.log("val/verb_loss", verb_loss)
        self.log("val/phase_loss", phase_loss)
        self.log("val/total_loss", total_loss)
        
        # Log task weights
        self.log("val/weight_iv", self.iv_weight)
        self.log("val/weight_tool", self.tool_weight)
        self.log("val/weight_verb", self.verb_weight)
        self.log("val/weight_phase", self.phase_weight)
        
        # Log normalized losses
        self.log("val/norm_iv_loss", norm_iv_loss)
        self.log("val/norm_tool_loss", norm_tool_loss)
        self.log("val/norm_verb_loss", norm_verb_loss)
        self.log("val/norm_phase_loss", norm_phase_loss)
        
        # Accumulate predictions and targets for metric computation
        self.predictions["iv"].append(iv_output.cpu().numpy())
        self.predictions["tools"].append(tool_output.cpu().numpy())
        self.predictions["verbs"].append(verb_output.cpu().numpy())
        self.predictions["phases"].append(phase_output.cpu().numpy())
        self.targets["iv"].append(iv_label.cpu().numpy())
        self.targets["tools"].append(tool_label.cpu().numpy())
        self.targets["verbs"].append(verb_label.cpu().numpy())
        self.targets["phases"].append(phase_label.cpu().numpy())
        
        return {
            "val_loss": total_loss,
            "iv_loss": iv_loss,
            "tool_loss": tool_loss,
            "verb_loss": verb_loss,
            "phase_loss": phase_loss,
            "norm_iv_loss": norm_iv_loss,
            "norm_tool_loss": norm_tool_loss,
            "norm_verb_loss": norm_verb_loss,
            "norm_phase_loss": norm_phase_loss
        }
    
    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch to compute metrics.
        
        Computes mean Average Precision (mAP) for all tasks and logs results.
        """
        # Compute mAP for each task
        mAP_iv = self.compute_mAP("iv")
        mAP_tools = self.compute_mAP("tools")
        mAP_verbs = self.compute_mAP("verbs")
        mAP_phases = self.compute_mAP("phases")
        
        # Log metrics
        self.log("val/mAP_iv", mAP_iv)
        self.log("val/mAP_tools", mAP_tools)
        self.log("val/mAP_verbs", mAP_verbs)
        self.log("val/mAP_phases", mAP_phases)
        
        # Reset accumulators for next epoch
        self.predictions = {"iv": [], "tools": [], "verbs": [], "phases": []}
        self.targets = {"iv": [], "tools": [], "verbs": [], "phases": []}
    
    def test_step(self, batch, batch_idx):
        """
        Test step logic. Uses the same procedure as validation step.
        """
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        """
        Called at the end of test epoch. Uses the same metric computation as validation.
        """
        self.on_validation_epoch_end()

    def compute_mAP(self, component):
        """
        Compute mean Average Precision for a specific component/task.
        
        Args:
            component (str): Name of the component to compute mAP for ('iv', 'tools', 'verbs', or 'phases')
            
        Returns:
            float: Mean Average Precision value
        """
        # Concatenate all predictions and targets for the component
        targets = np.concatenate(self.targets[component])
        predicts = np.concatenate(self.predictions[component])
        
        # Compute AP for each class and handle potential warnings
        with warnings.catch_warnings():
            # Compute classwise AP and handle NaN values
            classwise = average_precision_score(targets, predicts, average=None)
            classwise = np.array([0 if np.isnan(x) else x for x in classwise])
            mean = np.nanmean(classwise)
        
        # Log AP for each class
        for i, ap in enumerate(classwise):
            self.log(f"val/AP_{component}_{i}", ap)
        
        return mean

    def on_epoch_start(self):
        """
        Called at the start of each epoch.
        
        After 10 epochs, unfreeze the last two layers of the feature extractor
        to allow fine-tuning of deeper features.
        """
        if self.current_epoch == 10:
            for param in self.model.feature_extractor[-2:].parameters():
                param.requires_grad = True