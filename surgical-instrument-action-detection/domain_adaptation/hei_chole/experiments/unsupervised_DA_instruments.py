from dataloader import balanced_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import wandb
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        # YOLO Initialisierung nur für Feature Extraction
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.yolo_model.eval()
        self.feature_layer = 8  # Layer 8 für domain-invariante Features
        
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

    def forward(self, x, alpha=1.0):
        # Feature Extraction
        features = None
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
                    break
                    
        features = self.feature_reducer(features)
        
        # Domain Classification mit Gradient Reversal
        domain_features = GradientReversalLayer.apply(features, alpha)
        domain_pred = self.domain_classifier(domain_features)
        
        return domain_pred

def get_alpha(epoch, num_epochs=30):
    """Implementierung eines Warm-up Schedules für GRL"""
    p = epoch / num_epochs
    return 2. / (1. + np.exp(-10 * p)) - 1

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Tracking Metriken
    domain_accuracies = []
    source_accuracies = []
    target_accuracies = []
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        domains = batch['domain'].to(device)
        
        alpha = get_alpha(epoch)
        optimizer.zero_grad()
        
        # Nur Domain Classification
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
        
        # Berechne Metriken
        with torch.no_grad():
            domain_preds = (domain_pred.squeeze() > 0.5).float()
            accuracy = (domain_preds == domains).float().mean().item()
            
            # Separate Metriken für Source und Target
            source_mask = domains == 0
            target_mask = domains == 1
            
            if source_mask.any():
                source_acc = (domain_preds[source_mask] == domains[source_mask]).float().mean().item()
                source_accuracies.append(source_acc)
            
            if target_mask.any():
                target_acc = (domain_preds[target_mask] == domains[target_mask]).float().mean().item()
                target_accuracies.append(target_acc)
            
            domain_accuracies.append(accuracy)
        
        # Logging
        if batch_idx % 10 == 0:
            wandb.log({
                "batch_domain_loss": domain_loss.item(),
                "batch_domain_accuracy": accuracy,
                "alpha": alpha,
                "batch_source_ratio": (domains == 0).float().mean().item(),
                "batch_target_ratio": (domains == 1).float().mean().item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "domain_confusion_score": 1 - abs(2 * accuracy - 1)
            })
    
    # Epoch Statistiken
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
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    # Tracking Metriken
    domain_accuracies = []
    source_accuracies = []
    target_accuracies = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            domains = batch['domain'].to(device)
            
            alpha = get_alpha(epoch)
            domain_pred = model(images, alpha)
            
            # Domain Loss
            val_loss = F.binary_cross_entropy(
                domain_pred.squeeze().clamp(1e-7, 1),
                domains.float()
            )
            
            total_val_loss += val_loss.item()
            num_batches += 1
            
            # Berechne Metriken
            domain_preds = (domain_pred.squeeze() > 0.5).float()
            accuracy = (domain_preds == domains).float().mean().item()
            
            # Separate Metriken für Source und Target
            source_mask = domains == 0
            target_mask = domains == 1
            
            if source_mask.any():
                source_acc = (domain_preds[source_mask] == domains[source_mask]).float().mean().item()
                source_accuracies.append(source_acc)
            
            if target_mask.any():
                target_acc = (domain_preds[target_mask] == domains[target_mask]).float().mean().item()
                target_accuracies.append(target_acc)
            
            domain_accuracies.append(accuracy)
    
    # Validierungs Statistiken
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
    # Wandb Konfiguration
    config = {
        "learning_rate": 0.0003,  # Reduziert für stabileres Training
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
    
    # Dataloader
    train_loader = balanced_dataloader(split='train')
    val_loader = balanced_dataloader(split='val')
    
    # Model Initialization
    model = DomainAdapter(yolo_path=config["model_path"]).to(device)
    
    # Optimizer mit Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Training Setup
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    save_dir = Path("domain_adapter_weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training Loop
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, device, epoch)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Learning Rate Scheduling basierend auf Validierungs-Loss
        scheduler.step(val_loss)
        
        # Early Stopping und Model Saving basierend auf Domain Accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            save_feature_reducer(model, save_dir / 'best_feature_reducer.pt')
            
            wandb.log({
                "best_val_accuracy": val_acc,
                "best_val_loss": val_loss,
                "best_model_epoch": epoch
            })
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= config["patience"]:
            print(f"Early stopping triggered after epoch {epoch}")
            break
    
    wandb.finish()

if __name__ == "__main__":
    main()