import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
from dataloader import balanced_dataloader
import ultralytics.nn.modules.conv

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

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
    
class SpatialDomainAdapter(nn.Module):
    def __init__(self, yolo_path="/data/Bartscht/YOLO/best_v35.pt"):
        super().__init__()
        # YOLO Setup
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        self.feature_layer = 16
        
        # Freeze YOLO
        self.yolo_model.eval()
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        
        # Feature Reducer (Layer 16 hat 256 Kanäle)
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=32),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

    def set_train_mode(self, mode=True):
        self.feature_reducer.train(mode)
        self.domain_classifier.train(mode)
        return self

    def forward(self, x, alpha=1.0, return_features=False):
    # Feature Extraktion bis Layer 16
        features = []  # Liste für Zwischenfeatures
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                if isinstance(layer, ultralytics.nn.modules.conv.Concat):
                    # Concat-Layer korrekt behandeln
                    x = torch.cat([x] + features[-layer.d:], 1)
                else:
                    x = layer(x)
                
                features.append(x)
                
                if i == self.feature_layer:
                    extracted_features = x.clone()
                    break

        # Feature Reduction und Domain Adaptation
        reduced_features = self.feature_reducer(extracted_features)
        domain_features = GradientReversalLayer.apply(reduced_features, alpha)
        domain_pred = self.domain_classifier(domain_features)

        if return_features:
            return domain_pred, reduced_features
        return domain_pred

def calculate_spatial_metrics(domain_preds, domains):
    """Berechnet Metriken für räumliche Domain Predictions"""
    # Expand domains to match spatial dimensions
    B, _, H, W = domain_preds.shape
    domains = domains.view(B, 1, 1, 1).expand(-1, 1, H, W)
    
    # Calculate metrics across all spatial positions
    domain_preds = domain_preds.squeeze(1)  # Remove channel dim for calculations
    domains = domains.squeeze(1)
    
    accuracy = (domain_preds > 0.5).float().eq(domains).float().mean(dim=[1,2])
    
    source_mask = domains[:, 0, 0] == 0
    target_mask = domains[:, 0, 0] == 1
    
    source_acc = accuracy[source_mask].mean() if source_mask.any() else torch.tensor(0.)
    target_acc = accuracy[target_mask].mean() if target_mask.any() else torch.tensor(0.)
    
    domain_confusion = 1 - torch.abs(2 * accuracy.mean() - 1)
    
    return {
        'accuracy': accuracy.mean().item(),
        'source_acc': source_acc.item(),
        'target_acc': target_acc.item(),
        'domain_confusion': domain_confusion.item()
    }

def spatial_consistency_loss(source_features, target_features):
    """Berechnet Feature-Consistency über räumliche Dimensionen"""
    # Calculate mean source features while preserving spatial dimensions
    mean_source = source_features.mean(dim=0, keepdim=True)  # [1, C, H, W]
    
    # Compute similarity at each spatial location
    similarity = F.cosine_similarity(
        mean_source.expand(target_features.size(0), -1, -1, -1),
        target_features,
        dim=1  # Similarity over channel dimension
    ).mean([1, 2])  # Mean over spatial dimensions
    
    return similarity.mean()  # Mean over batch

def train_epoch(model, dataloader, optimizer, device, epoch, config):
    model.set_train_mode(True)
    metrics_tracker = {'loss': [], 'feature_sim': [], 'domain_metrics': []}
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{config["num_epochs"]}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        domains = batch['domain'].to(device)
        
        # Sanfter GRL-Effekt
        alpha = 0.2 * (2. / (1. + np.exp(-5 * epoch / config["num_epochs"])) - 1)
        
        optimizer.zero_grad()
        domain_pred, features = model(images, alpha, return_features=True)
        
        B, C, H, W = domain_pred.shape  # C sollte 1 sein
        expanded_domains = domains.view(B, 1, 1, 1).expand(B, 1, H, W)

        # Räumlicher Domain Classification Loss
        domain_loss = F.binary_cross_entropy(
            domain_pred,
            expanded_domains.float()
        )
        
        # Räumliche Feature Consistency
        source_features = features[domains == 0]
        target_features = features[domains == 1]
        
        if source_features.size(0) > 0 and target_features.size(0) > 0:
            similarity = spatial_consistency_loss(source_features, target_features)
            total_loss = domain_loss - config["feature_consistency_weight"] * similarity
            metrics_tracker['feature_sim'].append(similarity.item())
        else:
            total_loss = domain_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Tracking
        metrics_tracker['loss'].append(total_loss.item())
        metrics_tracker['domain_metrics'].append(
            calculate_spatial_metrics(domain_pred.detach(), domains)
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
    model.set_train_mode(False)
    metrics_tracker = {'loss': [], 'feature_sim': [], 'domain_metrics': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Val Epoch {epoch+1}'):
            images = batch['image'].to(device)
            domains = batch['domain'].to(device)
            
            domain_pred, features = model(images, alpha=0.2, return_features=True)
            
            # Korrekte Domain Expansion
            B, C, H, W = domain_pred.shape
            expanded_domains = domains.view(B, 1, 1, 1).expand(B, 1, H, W)
            
            # Domain Classification Loss
            domain_loss = F.binary_cross_entropy(
                domain_pred,  # Shape: [B, 1, H, W]
                expanded_domains.float()  # Shape: [B, 1, H, W]
            )
            
            # Räumliche Feature Consistency
            source_features = features[domains == 0]
            target_features = features[domains == 1]
            
            if source_features.size(0) > 0 and target_features.size(0) > 0:
                similarity = spatial_consistency_loss(source_features, target_features)
                metrics_tracker['feature_sim'].append(similarity.item())
            
            metrics_tracker['loss'].append(domain_loss.item())
            metrics_tracker['domain_metrics'].append(
                calculate_spatial_metrics(domain_pred, domains)
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
        'epoch': epoch,
        'spatial_info': {
            'input_shape': [512, None, None],  # H/16, W/16 werden dynamisch angepasst
            'output_shape': [256, None, None]  # H/16, W/16 werden dynamisch angepasst
        }
    }
    
    # Speichere komplettes Checkpoint
    torch.save(checkpoint, save_dir / f'spatial_model_epoch_{epoch}.pt')
    
    # Speichere nur Feature Reducer für einfache Weiterverwendung
    torch.save({
        'state_dict': model.feature_reducer.state_dict(),
        'spatial_info': checkpoint['spatial_info']
    }, save_dir / 'spatial_feature_reducer.pt')

def main():
    config = {
        "learning_rate": 0.0001,
        "num_epochs": 30,
        "batch_size": 32,
        "patience": 10,
        "model_path": "/data/Bartscht/YOLO/best_v35.pt",
        "feature_consistency_weight": 0.1,
        "min_lr": 1e-6,
        "experiment_name": "spatial_domain_adaptation"
    }
    
    wandb.init(
        project="surgical-domain-adaptation-unsupervised",
        name=config["experiment_name"],
        config=config,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataloaders
    train_loader = balanced_dataloader(split='train')
    val_loader = balanced_dataloader(split='val')
    
    # Model Setup
    model = SpatialDomainAdapter(yolo_path=config["model_path"]).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=config["min_lr"]
    )
    
    # Training Setup
    best_feature_sim = -float('inf')
    patience_counter = 0
    save_dir = Path(f"spatial_domain_adapter_weights_{config['experiment_name']}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting Spatial Feature-Focused Training...")
    
    try:
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
    
    except KeyboardInterrupt:
        print("\nTraining wurde manuell unterbrochen. Speichere letztes Modell...")
        save_model(model, val_metrics, epoch, save_dir)
    
    print("\nTraining abgeschlossen!")
    wandb.finish()

if __name__ == "__main__":
    main()