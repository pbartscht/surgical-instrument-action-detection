import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import logging
import numpy as np
from pathlib import Path
import time
from collections import defaultdict
from sklearn.metrics import average_precision_score, f1_score
from ultralytics import YOLO
from label_loader import SurgicalDataset, TOOL_MAPPING
import argparse
import wandb

CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING = {
    'grasper': 'grasper',
    'bipolar': 'coagulation',
    'clipper': 'clipper',
    'hook': 'coagulation',
    'scissors': 'scissors',
    'irrigator': 'suction_irrigation'
}

TOOL_MAPPING = {
    0: 'grasper', 
    1: 'bipolar', 
    2: 'hook',
    3: 'scissors', 
    4: 'clipper', 
    5: 'irrigator'
}


class FeatureAlignmentHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Instrument classifier
        self.instrument_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x, alpha=1.0):
        features = self.feature_reducer(x)
        
        # Apply gradient reversal for domain classification
        domain_pred = self.domain_classifier(
            GradientReversal.apply(features, alpha)
        )
        # Normal forward pass for instrument classification
        class_pred = self.instrument_classifier(features)
        
        return domain_pred, class_pred

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradients during backward pass
        return -ctx.alpha * grad_output, None

class DomainAdaptationTrainer:
    def __init__(self, yolo_path, device, save_dir, 
                 domain_lambda=0.3, 
                 alpha_schedule=5.0,
                 feature_layer=10):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.domain_lambda = domain_lambda  # Weight for domain loss
        self.alpha_schedule = alpha_schedule  # Controls adaptation speed
        self.feature_layer = feature_layer  # Which YOLO layer to get features from
        self.current_epoch = 0
        
        # Load and freeze YOLO model
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(device)
        self.yolo_model.model.eval()
        
        # Initialize alignment head
        self.alignment_head = FeatureAlignmentHead(
            num_classes=len(TOOL_MAPPING)
        ).to(device)
        
        # Optimizer setup
        self.optimizer = torch.optim.Adam(
            self.alignment_head.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            patience=5,
            factor=0.5
        )
        
        # Setup logging
        self.writer = SummaryWriter(log_dir=str(self.save_dir / 'runs'))
        setup_logging(self.save_dir)
        
        # Feature statistics tracking
        self.feature_stats = {
            'mean': [], 'std': [], 'grad_norm': []
        }
    
    def extract_features(self, images):
        """Extract features from specific YOLO layer"""
        images = images.to(self.device)
        x = images
        features = None
        
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model.model.model):
                layer = layer.to(self.device)
                x = layer(x)
                if i == self.feature_layer:
                    features = x
                    self.log_feature_stats(features, 'extract')
                    break
        
        return features
    
    def log_feature_stats(self, features, phase):
        """Log feature statistics for monitoring"""
        with torch.no_grad():
            mean = features.mean().item()
            std = features.std().item()
            grad_norm = features.grad.norm().item() if features.grad is not None else 0.0
            
            logging.info(f"{phase} Features - Mean: {mean:.4f}, Std: {std:.4f}, Grad: {grad_norm:.4f}")
            
            step = len(self.feature_stats['mean'])
            self.writer.add_scalar(f'Features/{phase}/mean', mean, step)
            self.writer.add_scalar(f'Features/{phase}/std', std, step)
            self.writer.add_scalar(f'Features/{phase}/grad_norm', grad_norm, step)
    
    def train_step(self, batch, epoch_progress):
        """Supervised Domain Adaptation Training Step"""
        self.alignment_head.train()
        self.optimizer.zero_grad()
        
        # Get batch data and ensure everything is on the device
        images = batch['image'].to(self.device)
        labels = batch['labels'].float().to(self.device)  # Ensure float
        domain_labels = batch['domain'].float().to(self.device)  # Ensure float
        
        # Trenne Source und Target Samples
        source_mask = domain_labels == 0
        target_mask = domain_labels == 1
        
        # Feature Extraction
        features = self.extract_features(images)
        
        # Alpha für GRL
        alpha = min(2. / (1. + np.exp(-self.alpha_schedule * epoch_progress)) - 1, 0.8)
        
        # Forward pass
        domain_pred, class_pred = self.alignment_head(features, alpha)
        
        # Ensure predictions are in valid range [0,1]
        class_pred = torch.clamp(class_pred, 0, 1)
        domain_pred = torch.clamp(domain_pred, 0, 1)
        
        # Classification Loss für Source Domain
        if source_mask.any():
            source_clf_loss = F.binary_cross_entropy(
                class_pred[source_mask],
                labels[source_mask],
                reduction='mean'
            )
        else:
            source_clf_loss = torch.tensor(0.0).to(self.device)
        
        # Spezieller Classification Loss für Target Domain mit Mapping
        if target_mask.any():
            target_clf_loss = self.compute_target_loss(
                class_pred[target_mask],
                labels[target_mask]
            )
        else:
            target_clf_loss = torch.tensor(0.0).to(self.device)
        
        # Kombinierter Classification Loss
        clf_loss = source_clf_loss + target_clf_loss
        
        # Domain Classification Loss
        domain_loss = F.binary_cross_entropy(
            domain_pred,
            domain_labels.unsqueeze(1),
            reduction='mean'
        )
        
        # Gesamtverlust
        total_loss = clf_loss + self.domain_lambda * domain_loss
        
        # Backward und Optimize
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.alignment_head.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'source_clf_loss': source_clf_loss.item(),
            'target_clf_loss': target_clf_loss.item(),
            'domain_loss': domain_loss.item(),
            'alpha': alpha
        }
    
    def compute_target_loss(self, pred, target):
        """Spezieller Loss für Target Domain mit Instrument-Mapping"""
        # Ensure inputs are in valid range
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Mapping-Matrix erstellen
        mapping_matrix = torch.zeros(len(TOOL_MAPPING), len(set(CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.values())))
        for source_idx, source_tool in TOOL_MAPPING.items():
            target_tool = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING[source_tool]
            target_idx = list(set(CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.values())).index(target_tool)
            mapping_matrix[source_idx, target_idx] = 1
        mapping_matrix = mapping_matrix.to(self.device)
        
        # Transformiere Predictions gemäß Mapping
        mapped_pred = torch.clamp(torch.matmul(pred, mapping_matrix), 0, 1)
        mapped_target = torch.clamp(torch.matmul(target, mapping_matrix), 0, 1)
        
        # Berechne Loss
        return F.binary_cross_entropy(mapped_pred, mapped_target, reduction='mean')
    
    def validate(self, val_loader):
        """Validierung mit Instrument-Mapping"""
        self.alignment_head.eval()
        metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                domain = batch['domain'].to(self.device)
                
                features = self.extract_features(images)
                _, predictions = self.alignment_head(features)
                
                # Separate Metriken für Source und Target
                for domain_type, mask in [('source', domain == 0), ('target', domain == 1)]:
                    if not mask.any():
                        continue
                        
                    domain_pred = predictions[mask]
                    domain_labels = labels[mask]
                    
                    if domain_type == 'target':
                        # Verwende Mapping für Target Domain
                        domain_pred = self.apply_instrument_mapping(domain_pred)
                        domain_labels = self.apply_instrument_mapping(domain_labels)
                    
                    # Berechne Metriken
                    for i, tool_name in enumerate(TOOL_MAPPING):
                        metrics[f'{domain_type}_ap_{tool_name}'].append(
                            average_precision_score(
                                domain_labels[:, i].cpu().numpy(),
                                domain_pred[:, i].cpu().numpy()
                            )
                        )
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint and return path"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.alignment_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        save_path = self.save_dir / f'alignment_head_epoch_{epoch}.pt'
        torch.save(checkpoint, save_path)
        logging.info(f"Checkpoint saved: {save_path}")
        
        return save_path  # Return the path for wandb logging

def setup_logging(save_dir):
    """Setup logging configuration"""
    log_file = Path(save_dir) / f'training_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train(args):
    """Main training function with WandB integration"""
    # Initialize wandb
    run = wandb.init(
        project="surgical-domain-adaptation",
        config=vars(args),
        name=f"domain-adapt_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Update save directory to use checkpoints folder with wandb run name
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = Path(__file__).parent / 'checkpoints' / run.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = DomainAdaptationTrainer(
        yolo_path=args.yolo_path,
        device=args.device,
        save_dir=save_dir,
        domain_lambda=args.domain_lambda,
        alpha_schedule=args.alpha_schedule
    )
    
    # Setup data loaders (same as before)
    source_dataset = SurgicalDataset(
        dataset_dir=args.source_dir,
        dataset_type='source'
    )
    target_train_dataset = SurgicalDataset(
        dataset_dir=Path(args.target_dir) / 'train',
        dataset_type='target'
    )
    target_val_dataset = SurgicalDataset(
        dataset_dir=Path(args.target_dir) / 'val',
        dataset_type='target'
    )
    
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    target_loader = DataLoader(
        target_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        target_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    best_val_metric = float('inf')
    
    # Training loop with wandb logging
    for epoch in range(args.num_epochs):
        trainer.current_epoch = epoch
        epoch_losses = []
        
        # Training steps
        for i, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
            progress = (epoch + i / len(source_loader)) / args.num_epochs
            
            combined_batch = {
                'image': torch.cat([source_batch['image'], target_batch['image']]),
                'labels': torch.cat([source_batch['labels'], target_batch['labels']]),
                'domain': torch.cat([
                    torch.zeros(len(source_batch['image'])),
                    torch.ones(len(target_batch['image']))
                ])
            }
            
            losses = trainer.train_step(combined_batch, progress)
            epoch_losses.append(losses)
            
            # Log to wandb more frequently
            if i % args.log_interval == 0:
                wandb.log({
                    'train/total_loss': losses['total_loss'],
                    'train/clf_loss': losses['clf_loss'],
                    'train/domain_loss': losses['domain_loss'],
                    'train/alpha': losses['alpha'],
                    'train/learning_rate': trainer.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': i
                })
                
                logging.info(
                    f"Epoch [{epoch}/{args.num_epochs}], "
                    f"Step [{i}/{len(source_loader)}], "
                    f"Loss: {losses['total_loss']:.4f}, "
                    f"CLF Loss: {losses['clf_loss']:.4f}, "
                    f"Domain Loss: {losses['domain_loss']:.4f}, "
                    f"Alpha: {losses['alpha']:.4f}"
                )
        
        # Calculate and log average epoch metrics
        avg_losses = {
            k: np.mean([d[k] for d in epoch_losses])
            for k in epoch_losses[0].keys()
        }
        
        wandb.log({
            'epoch/avg_total_loss': avg_losses['total_loss'],
            'epoch/avg_clf_loss': avg_losses['clf_loss'],
            'epoch/avg_domain_loss': avg_losses['domain_loss'],
            'epoch/avg_alpha': avg_losses['alpha'],
            'epoch': epoch
        })
        
        # Validation phase
        val_metrics = trainer.validate(val_loader)
        
        # Log validation metrics
        wandb.log({
            'val/ap': val_metrics['ap_0'],
            **{f'val/{k}': v for k, v in val_metrics.items()},
            'epoch': epoch
        })
        
        # Update learning rate
        trainer.scheduler.step(val_metrics['ap_0'])
        
        # Save best model and log to wandb
        if val_metrics['ap_0'] > best_val_metric:
            best_val_metric = val_metrics['ap_0']
            checkpoint_path = trainer.save_checkpoint(epoch, {**avg_losses, **val_metrics})
            wandb.save(str(checkpoint_path))  # Log checkpoint file to wandb
            
            wandb.log({
                'best_val_ap': best_val_metric,
                'epoch': epoch
            })
        
        # Log epoch summary
        logging.info(
            f"\nEpoch {epoch} Summary:\n"
            f"Average Training Loss: {avg_losses['total_loss']:.4f}\n"
            f"Validation AP: {val_metrics['ap_0']:.4f}\n"
            f"Best Validation AP: {best_val_metric:.4f}\n"
        )
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add existing arguments
    parser.add_argument('--yolo_path', type=str, required=True,
                      help='Path to pretrained YOLO model')
    parser.add_argument('--source_dir', type=str, required=True,
                      help='Path to CholecT50 dataset')
    parser.add_argument('--target_dir', type=str, required=True,
                      help='Path to HeiChole domain_adaptation directory')
    parser.add_argument('--save_dir', type=str, default=f'run_{time.strftime("%Y%m%d_%H%M%S")}',
                      help='Directory name for results (will be created under checkpoints/)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--domain_lambda', type=float, default=0.3,
                      help='Weight for domain adaptation loss')
    parser.add_argument('--alpha_schedule', type=float, default=5.0,
                      help='Speed of domain adaptation')
    parser.add_argument('--log_interval', type=int, default=10,
                      help='How often to log training progress')
    
    # Add new wandb-specific arguments
    parser.add_argument('--wandb_project', type=str, default='surgical-domain-adaptation',
                      help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='WandB entity/username')
    
    args = parser.parse_args()
    train(args)