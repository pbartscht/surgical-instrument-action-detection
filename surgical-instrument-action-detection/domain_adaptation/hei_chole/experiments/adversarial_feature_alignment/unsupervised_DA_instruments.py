from dataloader import balanced_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import wandb
from pathlib import Path
from torch.optim import lr_scheduler
from tqdm import tqdm

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainAdapter(nn.Module):
    def __init__(self, yolo_path="/data/Bartscht/YOLO/best_v35.pt"):
        super().__init__()
        # YOLO Initialization for feature extraction only
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        self.feature_layer = 8  # Layer 8 for domain-invariant features
        
        # Disable training for YOLO layers
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.yolo_model.eval()  # Always keep YOLO in eval mode
        
        # Feature Reducer
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5)
        )

        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def set_train_mode(self, mode=True):
        """Custom method to set training mode for feature_reducer and domain_classifier"""
        self.feature_reducer.train(mode)
        self.domain_classifier.train(mode)
        return self

    def forward(self, x, alpha=1.0):
        # Feature Extraction (always in eval mode)
        features = None
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
                    break
                    
        features = self.feature_reducer(features)
        
        # Domain Classification with Gradient Reversal
        domain_features = GradientReversalLayer.apply(features, alpha)
        domain_pred = self.domain_classifier(domain_features)
        
        return domain_pred

def get_alpha(epoch, num_epochs=30):
    """Implementierung eines Warm-up Schedules für GRL"""
    p = epoch / num_epochs
    return 2. / (1. + np.exp(-10 * p)) - 1

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.set_train_mode(True)
    total_loss = 0
    num_batches = 0
    
    # Tracking metrics
    domain_accuracies = []
    source_accuracies = []
    target_accuracies = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch+1}', 
                leave=True, position=0)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        domains = batch['domain'].to(device)
        
        alpha = get_alpha(epoch)
        optimizer.zero_grad()
        
        # Domain Classification
        domain_pred = model(images, alpha)
        
        # Domain Loss
        domain_loss = F.binary_cross_entropy(
            domain_pred.squeeze().clamp(1e-7, 1),
            domains.float()
        )
        
        domain_loss.backward()
        optimizer.step()
        
        total_loss += domain_loss.item()
        num_batches += 1
        
        # Calculate metrics
        with torch.no_grad():
            domain_preds = (domain_pred.squeeze() > 0.5).float()
            accuracy = (domain_preds == domains).float().mean().item()
            
            source_mask = domains == 0
            target_mask = domains == 1
            
            if source_mask.any():
                source_acc = (domain_preds[source_mask] == domains[source_mask]).float().mean().item()
                source_accuracies.append(source_acc)
            
            if target_mask.any():
                target_acc = (domain_preds[target_mask] == domains[target_mask]).float().mean().item()
                target_accuracies.append(target_acc)
            
            domain_accuracies.append(accuracy)
        
        # Update progress bar description
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f'Epoch {epoch+1} - Loss: {domain_loss.item():.4f}, Acc: {accuracy:.4f}, LR: {current_lr:.6f}')
        
        # Logging every 10 batches
        if batch_idx % 10 == 0:
            wandb.log({
                "batch_domain_loss": domain_loss.item(),
                "batch_domain_accuracy": accuracy,
                "alpha": alpha,
                "batch_source_ratio": (domains == 0).float().mean().item(),
                "batch_target_ratio": (domains == 1).float().mean().item(),
                "learning_rate": current_lr,
                "domain_confusion_score": 1 - abs(2 * accuracy - 1)
            })
    
    # Epoch statistics
    avg_loss = total_loss / num_batches
    avg_domain_acc = np.mean(domain_accuracies)
    avg_source_acc = np.mean(source_accuracies) if source_accuracies else 0
    avg_target_acc = np.mean(target_accuracies) if target_accuracies else 0
    
    wandb.log({
        "epoch": epoch,
        "train_domain_loss": avg_loss,
        "train_domain_accuracy": avg_domain_acc,
        "train_source_accuracy": avg_source_acc,
        "train_target_accuracy": avg_target_acc
    })
    
    return avg_loss, avg_domain_acc

def validate_epoch(model, val_loader, device, epoch):
    model.set_train_mode(False)
    total_val_loss = 0
    num_batches = 0
    
    domain_accuracies = []
    source_accuracies = []
    target_accuracies = []
    
    # Progress bar for validation
    pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}', 
                leave=True, position=0)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            domains = batch['domain'].to(device)
            
            alpha = get_alpha(epoch)
            domain_pred = model(images, alpha)
            
            val_loss = F.binary_cross_entropy(
                domain_pred.squeeze().clamp(1e-7, 1),
                domains.float()
            )
            
            total_val_loss += val_loss.item()
            num_batches += 1
            
            domain_preds = (domain_pred.squeeze() > 0.5).float()
            accuracy = (domain_preds == domains).float().mean().item()
            
            source_mask = domains == 0
            target_mask = domains == 1
            
            if source_mask.any():
                source_acc = (domain_preds[source_mask] == domains[source_mask]).float().mean().item()
                source_accuracies.append(source_acc)
            
            if target_mask.any():
                target_acc = (domain_preds[target_mask] == domains[target_mask]).float().mean().item()
                target_accuracies.append(target_acc)
            
            domain_accuracies.append(accuracy)
            
            # Update progress bar description
            pbar.set_description(f'Validation Epoch {epoch+1} - Loss: {val_loss.item():.4f}, Acc: {accuracy:.4f}')
    
    avg_val_loss = total_val_loss / num_batches
    avg_domain_acc = np.mean(domain_accuracies)
    avg_source_acc = np.mean(source_accuracies) if source_accuracies else 0
    avg_target_acc = np.mean(target_accuracies) if target_accuracies else 0
    
    wandb.log({
        "val_domain_loss": avg_val_loss,
        "val_domain_accuracy": avg_domain_acc,
        "val_source_accuracy": avg_source_acc,
        "val_target_accuracy": avg_target_acc,
        "val_epoch": epoch
    })
    
    return avg_val_loss, avg_domain_acc

def save_feature_reducer(model, path):
    """Speichert nur den Feature Reducer und relevante Informationen"""
    torch.save({
        'feature_reducer': model.feature_reducer.state_dict(),
        'feature_layer': model.feature_layer
    }, path)

def main():
    # Wandb Configuration
    config = {
        "learning_rate": 0.0003,
        "num_epochs": 30,
        "batch_size": 32,
        "patience": 7,
        "model_path": "/data/Bartscht/YOLO/best_v35.pt"
    }
    
    wandb.init(
        project="surgical-domain-adaptation-unsupervised",
        config=config
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader = balanced_dataloader(split='train')
    val_loader = balanced_dataloader(split='val')
    
    model = DomainAdapter(yolo_path=config["model_path"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Scheduler without verbose parameter
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    save_dir = Path("domain_adapter_weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training...")
    print(f"Total epochs: {config['num_epochs']}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_acc = validate_epoch(model, val_loader, device, epoch)
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nLearning rate: {current_lr:.6f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            save_feature_reducer(model, save_dir / 'best_feature_reducer.pt')
            print(f"\nSaved new best model (val_acc: {val_acc:.4f})")
            
            wandb.log({
                "best_val_accuracy": val_acc,
                "best_val_loss": val_loss,
                "best_model_epoch": epoch
            })
        else:
            patience_counter += 1
            
        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            break
    
    print("\nTraining completed!")
    wandb.finish()

if __name__ == "__main__":
    main()