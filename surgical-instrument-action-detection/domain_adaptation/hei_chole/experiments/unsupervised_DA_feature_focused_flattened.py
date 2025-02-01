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
        # YOLO Initialization
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        self.feature_layer = 8
        
        # Disable YOLO training
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.yolo_model.eval()
        
        # Feature Reducer mit Residual Connections
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, 3, padding=1, groups=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.consistency_loss = nn.CosineSimilarity(dim=1)

    def set_train_mode(self, mode=True):
        """Custom method to set training mode"""
        self.feature_reducer.train(mode)
        self.domain_classifier.train(mode)
        return self

    def forward(self, x, alpha=1.0, return_features=False):
        features = None
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
                    break
        
        reduced_features = self.feature_reducer(features)
        domain_features = GradientReversalLayer.apply(reduced_features, alpha)
        domain_pred = self.domain_classifier(domain_features)
        
        if return_features:
            return domain_pred, reduced_features
        return domain_pred

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

def calculate_metrics(domain_preds, domains):
    """Berechnet detaillierte Metriken für Domain Adaptation"""
    domain_preds = domain_preds.squeeze()
    accuracy = (domain_preds > 0.5).float().eq(domains).float().mean()
    
    source_mask = domains == 0
    target_mask = domains == 1
    
    source_acc = (domain_preds[source_mask] > 0.5).float().eq(domains[source_mask]).float().mean() if source_mask.any() else torch.tensor(0.)
    target_acc = (domain_preds[target_mask] > 0.5).float().eq(domains[target_mask]).float().mean() if target_mask.any() else torch.tensor(0.)
    
    domain_confusion = 1 - torch.abs(2 * accuracy - 1)
    
    return {
        'accuracy': accuracy.item(),
        'source_acc': source_acc.item(),
        'target_acc': target_acc.item(),
        'domain_confusion': domain_confusion.item()
    }

def train_epoch(model, dataloader, optimizer, device, epoch, config):
    model.set_train_mode(True)  # Nutze custom training mode
    metrics_tracker = {'loss': [], 'feature_sim': [], 'domain_metrics': []}
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{config["num_epochs"]}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        domains = batch['domain'].to(device)
        
        # Sanfter GRL-Effekt
        alpha = 0.2 * (2. / (1. + np.exp(-5 * epoch / config["num_epochs"])) - 1)
        
        optimizer.zero_grad()
        domain_pred, features = model(images, alpha, return_features=True)
        
        # Domain Classification Loss
        domain_loss = F.binary_cross_entropy(
            domain_pred.squeeze(),
            domains.float()
        )
        
        # Feature Consistency
        source_features = features[domains == 0]
        target_features = features[domains == 1]
        
        if source_features.size(0) > 0 and target_features.size(0) > 0:
            similarity = model.consistency_loss(
                source_features.mean(0, keepdim=True).expand(target_features.size(0), -1),
                target_features
            ).mean()
            
            total_loss = domain_loss - config["feature_consistency_weight"] * similarity
            metrics_tracker['feature_sim'].append(similarity.item())
        else:
            total_loss = domain_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        metrics_tracker['loss'].append(total_loss.item())
        metrics_tracker['domain_metrics'].append(
            calculate_metrics(domain_pred.detach(), domains)
        )
        
        if batch_idx % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = np.mean(metrics_tracker['loss'][-50:])
            avg_acc = np.mean([m['accuracy'] for m in metrics_tracker['domain_metrics'][-50:]])
            pbar.set_description(
                f'Epoch {epoch+1} - Loss: {avg_loss:.4f}, '
                f'Acc: {avg_acc:.4f}, LR: {current_lr:.6f}'
            )
            
            wandb.log({
                'batch/loss': total_loss.item(),
                'batch/domain_accuracy': metrics_tracker['domain_metrics'][-1]['accuracy'],
                'batch/feature_similarity': similarity.item() if 'similarity' in locals() else 0,
                'batch/alpha': alpha,
                'batch/learning_rate': current_lr
            })
    
    epoch_metrics = {
        'loss': np.mean(metrics_tracker['loss']),
        'feature_similarity': np.mean(metrics_tracker['feature_sim']) if metrics_tracker['feature_sim'] else 0,
        'domain_accuracy': np.mean([m['accuracy'] for m in metrics_tracker['domain_metrics']]),
        'source_accuracy': np.mean([m['source_acc'] for m in metrics_tracker['domain_metrics']]),
        'target_accuracy': np.mean([m['target_acc'] for m in metrics_tracker['domain_metrics']]),
        'domain_confusion': np.mean([m['domain_confusion'] for m in metrics_tracker['domain_metrics']])
    }
    
    return epoch_metrics

def validate_epoch(model, dataloader, device, epoch, config):
    model.set_train_mode(False)  # Hier die Änderung von eval() zu set_train_mode(False)
    metrics_tracker = {'loss': [], 'feature_sim': [], 'domain_metrics': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Val Epoch {epoch+1}'):
            images = batch['image'].to(device)
            domains = batch['domain'].to(device)
            
            domain_pred, features = model(images, alpha=0.2, return_features=True)
            
            # Metrics berechnen
            domain_loss = F.binary_cross_entropy(
                domain_pred.squeeze(),
                domains.float()
            )
            
            source_features = features[domains == 0]
            target_features = features[domains == 1]
            
            if source_features.size(0) > 0 and target_features.size(0) > 0:
                similarity = model.consistency_loss(
                    source_features.mean(0, keepdim=True).expand(target_features.size(0), -1),
                    target_features
                ).mean()
                metrics_tracker['feature_sim'].append(similarity.item())
            
            metrics_tracker['loss'].append(domain_loss.item())
            metrics_tracker['domain_metrics'].append(
                calculate_metrics(domain_pred, domains)
            )
    
    # Validation Metrics
    val_metrics = {
        'loss': np.mean(metrics_tracker['loss']),
        'feature_similarity': np.mean(metrics_tracker['feature_sim']) if metrics_tracker['feature_sim'] else 0,
        'domain_accuracy': np.mean([m['accuracy'] for m in metrics_tracker['domain_metrics']]),
        'source_accuracy': np.mean([m['source_acc'] for m in metrics_tracker['domain_metrics']]),
        'target_accuracy': np.mean([m['target_acc'] for m in metrics_tracker['domain_metrics']]),
        'domain_confusion': np.mean([m['domain_confusion'] for m in metrics_tracker['domain_metrics']])
    }
    
    return val_metrics

def save_model(model, metrics, epoch, save_dir):
    """Speichert Model-Checkpoints mit relevanten Metriken"""
    checkpoint = {
        'feature_reducer': model.feature_reducer.state_dict(),
        'domain_classifier': model.domain_classifier.state_dict(),
        'metrics': metrics,
        'epoch': epoch
    }
    torch.save(checkpoint, save_dir / f'model_epoch_{epoch}.pt')

def main():
    config = {
        "learning_rate": 0.0001,
        "num_epochs": 30,
        "batch_size": 32,
        "patience": 10,
        "model_path": "/data/Bartscht/YOLO/best_v35.pt",
        "feature_consistency_weight": 0.1,
        "min_lr": 1e-6
    }
    
    wandb.init(
        project="surgical-domain-adaptation-unsupervised",
        config=config,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataloaders
    train_loader = balanced_dataloader(split='train')
    val_loader = balanced_dataloader(split='val')
    
    # Model Setup
    model = DomainAdapter(yolo_path=config["model_path"]).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01
    )
    
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=config["min_lr"]
    )
    
    # Training Setup
    best_feature_sim = -float('inf')
    patience_counter = 0
    save_dir = Path("domain_adapter_weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting Feature-Focused Training...")
    
    for epoch in range(config["num_epochs"]):
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config)
        val_metrics = validate_epoch(model, val_loader, device, epoch, config)
        
        # Scheduler Step
        scheduler.step()
        
        # Logging
        wandb.log({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Model Saving basierend auf Feature-Similarity
        if val_metrics['feature_similarity'] > best_feature_sim:
            best_feature_sim = val_metrics['feature_similarity']
            save_model(model, val_metrics, epoch, save_dir)
            patience_counter = 0
            print(f"\nNeues bestes Modell gespeichert (Feature Similarity: {best_feature_sim:.4f})")
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= config["patience"]:
            print(f"\nEarly Stopping nach Epoch {epoch+1}")
            break
        
        # Epoch Summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Feature Sim: {train_metrics['feature_similarity']:.4f}, "
              f"Domain Acc: {train_metrics['domain_accuracy']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"Feature Sim: {val_metrics['feature_similarity']:.4f}, "
              f"Domain Acc: {val_metrics['domain_accuracy']:.4f}")
    
    print("\nTraining abgeschlossen!")
    wandb.finish()

if __name__ == "__main__":
    main()