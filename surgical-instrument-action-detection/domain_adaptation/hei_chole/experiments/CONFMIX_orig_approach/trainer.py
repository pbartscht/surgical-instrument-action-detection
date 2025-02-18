import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from ultralytics import YOLO

class ConfMixTrainer:
    def __init__(
        self,
        model_path,
        source_path,
        target_path,
        save_dir,
        img_size=640,
        batch_size=8,
        num_epochs=50,
        device='cuda'
    ):
        """
        ConfMix Trainer for Domain Adaptation
        
        Args:
            model_path: Path to pretrained YOLOv11 weights
            source_path: Path to source domain data
            target_path: Path to target domain data (video structure)
            save_dir: Directory to save results
            img_size: Input image size
            batch_size: Batch size for training
            num_epochs: Number of epochs to train
            device: Device to use for training
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_epochs = num_epochs
        
        # Load model
        print(f'Loading model from {model_path}...')
        self.model = YOLO(model_path)
        
        # Initialize dataloaders
        print('Creating dataloaders...')
        from dataset_loader import create_uda_dataloaders
        self.train_loader_s, self.dataset_s, self.train_loader_t, self.dataset_t = create_uda_dataloaders(
            source_path=source_path,
            target_path=target_path,
            img_size=img_size,
            batch_size=batch_size,
            augment=True
        )
        
        # Initialize ConfMix core
        from confmix_core import ConfMixCore
        self.confmix = ConfMixCore(self.model, device=device)
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.0001
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(str(self.save_dir / 'logs'))
        
        # Initialize best metrics
        self.best_map = 0.0

    def train(self):
        """Main training loop"""
        print(f'Starting training for {self.num_epochs} epochs...')
        total_iterations = len(self.train_loader_s) * self.num_epochs
        current_iteration = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_det_loss = 0
            epoch_conf_loss = 0
            
            # Create progress bar
            pbar = tqdm(
                zip(self.train_loader_s, self.train_loader_t),
                total=min(len(self.train_loader_s), len(self.train_loader_t)),
                desc=f'Epoch {epoch+1}/{self.num_epochs}'
            )
            
            # Training loop
            for batch_idx, (source_batch, target_batch) in enumerate(pbar):
                try:
                    # Forward pass through ConfMix
                    confmix_output = self.confmix.forward_step(
                        source_batch,
                        target_batch,
                        current_iteration,
                        total_iterations
                    )
                    
                    # Calculate losses
                    det_loss = self._compute_detection_loss(
                        confmix_output['source_pred'],
                        source_batch[1],  # source labels
                        confmix_output['source_var']
                    )
                    
                    conf_loss = self._compute_consistency_loss(
                        confmix_output['mixed_images'],
                        confmix_output['mixed_targets'],
                        confmix_output['mixing_masks']
                    )
                    
                    # Combine losses with gamma weighting
                    gamma = confmix_output['gamma']
                    total_loss = det_loss + gamma * conf_loss
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    epoch_loss += total_loss.item()
                    epoch_det_loss += det_loss.item()
                    epoch_conf_loss += conf_loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{total_loss.item():.4f}',
                        'det_loss': f'{det_loss.item():.4f}',
                        'conf_loss': f'{conf_loss.item():.4f}',
                        'gamma': f'{gamma.item():.4f}'
                    })
                    
                    current_iteration += 1
                    
                except Exception as e:
                    print(f'Error in batch {batch_idx}: {e}')
                    continue
            
            # End of epoch
            avg_loss = epoch_loss / len(self.train_loader_s)
            avg_det_loss = epoch_det_loss / len(self.train_loader_s)
            avg_conf_loss = epoch_conf_loss / len(self.train_loader_s)
            
            # Log metrics
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('Loss/detection', avg_det_loss, epoch)
            self.writer.add_scalar('Loss/consistency', avg_conf_loss, epoch)
            
            # Validation
            if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
                map50 = self._validate()
                self.writer.add_scalar('mAP50/val', map50, epoch)
                
                # Save best model
                if map50 > self.best_map:
                    self.best_map = map50
                    self._save_checkpoint(
                        epoch,
                        map50,
                        filename='best.pt'
                    )
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:  # Save every 10 epochs
                self._save_checkpoint(
                    epoch,
                    map50 if (epoch + 1) % 5 == 0 else None,
                    filename=f'epoch_{epoch+1}.pt'
                )
            
            # Update scheduler
            self.scheduler.step()
            
        # End of training
        self.writer.close()
        print('Training completed!')

    def _compute_detection_loss(self, predictions, targets, variances):
        """Compute detection loss for source domain"""
        # Use YOLO's built-in loss computation
        loss = self.model.criterion(predictions, targets, variances)
        return loss
    
    def _compute_consistency_loss(self, mixed_images, mixed_targets, mixing_masks):
        """Compute consistency loss for mixed samples"""
        # Forward pass on mixed images
        mixed_pred = self.model(mixed_images)
        
        # Only consider predictions in mixed region
        masked_pred = mixed_pred * mixing_masks.unsqueeze(1)
        masked_targets = mixed_targets * mixing_masks.unsqueeze(1)
        
        # Compute loss
        loss = nn.MSELoss()(masked_pred, masked_targets)
        return loss
    
    def _validate(self):
        """Run validation on target domain"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for imgs, targets, paths, _ in tqdm(self.train_loader_t, desc='Validating'):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Run inference
                pred = self.model(imgs)
                
                # Compute metrics
                results.append(
                    self.model.compute_metrics(pred, targets)
                )
        
        # Compute mean metrics
        metrics = {}
        for key in results[0].keys():
            metrics[key] = sum(r[key] for r in results) / len(results)
        
        return metrics['mAP50']
    
    def _save_checkpoint(self, epoch, map50=None, filename='checkpoint.pt'):
        """Save model checkpoint"""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
        }
        if map50 is not None:
            ckpt['map50'] = map50
            
        save_path = self.save_dir / filename
        torch.save(ckpt, save_path)
        print(f'Saved checkpoint to {save_path}')