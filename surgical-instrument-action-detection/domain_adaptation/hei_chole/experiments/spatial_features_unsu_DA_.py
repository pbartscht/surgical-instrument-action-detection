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
        self.feature_layer = 9  # Änderung: jetzt Layer 9 statt 16
        
        # Freeze YOLO
        self.yolo_model.eval()
        for param in self.yolo_model.parameters():
            param.requires_grad = False
            
        # Feature Reducer - angepasst für 512 Kanäle (Layer 9 Output)
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=32),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Domain Classifier - angepasst für 512 Kanäle Input
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

    def set_train_mode(self, mode=True):
        self.feature_reducer.train(mode)
        self.domain_classifier.train(mode)
        self.yolo_model.eval()
        return self

    def forward(self, x, alpha=1.0, return_features=False):
        # Extrahiere Features bis NACH Layer 9
        features = None
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                if i > self.feature_layer:
                    break
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
        
        # Wende Feature Reducer an
        reduced_features = self.feature_reducer(features)
        
        # Domain Classification mit Gradient Reversal
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
    # Features sollten die gleiche Größe haben wie der Input zu Layer 17
    mean_source = source_features.mean(dim=0, keepdim=True)
    
    # Ähnlichkeit über Kanaldimension berechnen
    similarity = F.cosine_similarity(
        mean_source.expand(target_features.size(0), -1, -1, -1),
        target_features,
        dim=1
    ).mean()
    
    return similarity

def train_epoch(model, dataloader, optimizer_reducer, optimizer_classifier, device, epoch, config):
    model.set_train_mode(True)
    # Neue Metrik-Struktur für den Wettkampf
    metrics_tracker = {
        'critic_loss': [],      # Erfolg des Kritikers (Domain Classifier)
        'artist_loss': [],      # Erfolg des Künstlers (Feature Reducer)
        'feature_sim': [],      # Bleibt von vorher - Feature Qualität
        'domain_metrics': []    # Bleibt von vorher - Detaillierte Metriken
    }
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{config["num_epochs"]}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        domains = batch['domain'].to(device)
        B, _, H, W = images.shape
        expanded_domains = domains.view(B, 1, 1, 1).expand(B, 1, H, W)

        # 1. Training des Kritikers (Domain Classifier)
        optimizer_classifier.zero_grad()
        domain_pred, reduced_features = model(images, alpha=1.0, return_features=True)
        critic_loss = F.binary_cross_entropy(domain_pred, expanded_domains.float())
        critic_loss.backward()
        optimizer_classifier.step()
        
        # 2. Training des Künstlers (Feature Reducer)
        optimizer_reducer.zero_grad()
        domain_pred, reduced_features = model(images, alpha=1.0, return_features=True)
        
        # Feature Consistency bleibt wie vorher
        source_features = reduced_features[domains == 0]
        target_features = reduced_features[domains == 1]
        
        if source_features.size(0) > 0 and target_features.size(0) > 0:
            similarity = spatial_consistency_loss(source_features, target_features)
            # Künstler will Kritiker täuschen (negatives Loss) und Features erhalten
            artist_loss = -F.binary_cross_entropy(domain_pred, expanded_domains.float()) + \
                         config["feature_consistency_weight"] * similarity
            metrics_tracker['feature_sim'].append(similarity.item())
        else:
            # Wenn nur eine Domain vorhanden, fokussiere auf Täuschung
            artist_loss = -F.binary_cross_entropy(domain_pred, expanded_domains.float())
        
        artist_loss.backward()
        optimizer_reducer.step()
        
        # Metrics Tracking
        metrics_tracker['critic_loss'].append(critic_loss.item())
        metrics_tracker['artist_loss'].append(-artist_loss.item())  # Negativ für intuitivere Metrik
        metrics_tracker['domain_metrics'].append(
            calculate_spatial_metrics(domain_pred.detach(), domains)
        )
        
        # Logging angepasst für beide Akteure
        if batch_idx % 10 == 0:
            critic_avg = np.mean(metrics_tracker['critic_loss'][-50:])
            artist_avg = np.mean(metrics_tracker['artist_loss'][-50:])
            pbar.set_description(
                f'Epoch {epoch+1} - Critic: {critic_avg:.4f}, Artist: {artist_avg:.4f}'
            )
    
    # Return-Format angepasst für beide Akteure
    return {
        'critic_loss': np.mean(metrics_tracker['critic_loss']),
        'artist_loss': np.mean(metrics_tracker['artist_loss']),
        'feature_similarity': np.mean(metrics_tracker['feature_sim']) if metrics_tracker['feature_sim'] else 0,
        'domain_accuracy': np.mean([m['accuracy'] for m in metrics_tracker['domain_metrics']]),
        'source_accuracy': np.mean([m['source_acc'] for m in metrics_tracker['domain_metrics']]),
        'target_accuracy': np.mean([m['target_acc'] for m in metrics_tracker['domain_metrics']]),
        'domain_confusion': np.mean([m['domain_confusion'] for m in metrics_tracker['domain_metrics']])
    }

def validate_epoch(model, dataloader, device, epoch, config):
    model.set_train_mode(False)
    metrics_tracker = {
        'loss': [], 
        'critic_loss': [],  # Hinzugefügt
        'artist_loss': [],  # Hinzugefügt
        'feature_sim': [], 
        'domain_metrics': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Val Epoch {epoch+1}'):
            images = batch['image'].to(device)
            domains = batch['domain'].to(device)
            
            # Forward Pass mit reduzierten Features
            domain_pred, reduced_features = model(images, alpha=0.2, return_features=True)
            
            # Domain Classification Loss für Kritiker
            B, C, H, W = domain_pred.shape
            expanded_domains = domains.view(B, 1, 1, 1).expand(B, 1, H, W)
            critic_loss = F.binary_cross_entropy(
                domain_pred,  # Shape: [B, 1, H, W]
                expanded_domains.float()  # Shape: [B, 1, H, W]
            )
            
            # Feature Consistency Check für Künstler
            source_features = reduced_features[domains == 0]
            target_features = reduced_features[domains == 1]
            
            if source_features.size(0) > 0 and target_features.size(0) > 0:
                similarity = spatial_consistency_loss(source_features, target_features)
                metrics_tracker['feature_sim'].append(similarity.item())
                
                # Künstler Loss - Täuschung des Kritikers und Feature Konsistenz
                artist_loss = -F.binary_cross_entropy(domain_pred, expanded_domains.float()) + \
                             config.get("feature_consistency_weight", 0.1) * similarity
                metrics_tracker['artist_loss'].append(artist_loss.item())
            else:
                # Wenn nur eine Domain vorhanden, fokussiere auf Täuschung
                artist_loss = -F.binary_cross_entropy(domain_pred, expanded_domains.float())
                metrics_tracker['artist_loss'].append(artist_loss.item())
            
            # Metrics Tracking
            metrics_tracker['loss'].append(critic_loss.item())
            metrics_tracker['critic_loss'].append(critic_loss.item())
            metrics_tracker['domain_metrics'].append(
                calculate_spatial_metrics(domain_pred, domains)
            )
    
    # Berechne Validierungs-Metriken mit zusätzlichen Informationen
    val_metrics = {
        'loss': np.mean(metrics_tracker['loss']),
        'critic_loss': np.mean(metrics_tracker['critic_loss']),
        'artist_loss': np.mean(metrics_tracker['artist_loss']),
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
    
    optimizer_reducer = torch.optim.AdamW(
    model.feature_reducer.parameters(),
    lr=config["learning_rate"],
    weight_decay=0.01
)
    optimizer_classifier = torch.optim.AdamW(
        model.domain_classifier.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01
    )

    # Scheduler für beide Optimierer
    scheduler_reducer = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_reducer,
        T_0=10,
        T_mult=2,
        eta_min=config["min_lr"]
    )
    scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_classifier,
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
            # Training mit neuen Optimizern
            train_metrics = train_epoch(
                model, 
                train_loader, 
                optimizer_reducer, 
                optimizer_classifier, 
                device, 
                epoch, 
                config
            )
            val_metrics = validate_epoch(model, val_loader, device, epoch, config)
            
            # Scheduler Step für beide Optimierer
            scheduler_reducer.step()
            scheduler_classifier.step()
            
            # Logging
            wandb.log({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'learning_rate_reducer': optimizer_reducer.param_groups[0]['lr'],
                'learning_rate_classifier': optimizer_classifier.param_groups[0]['lr']
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
            print(f"Train - Critic Loss: {train_metrics['critic_loss']:.4f}, "
                f"Artist Loss: {train_metrics['artist_loss']:.4f}, "
                f"Feature Sim: {train_metrics['feature_similarity']:.4f}, "
                f"Domain Acc: {train_metrics['domain_accuracy']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Critic Loss: {val_metrics['critic_loss']:.4f}, "
                f"Artist Loss: {val_metrics['artist_loss']:.4f}, "
                f"Feature Sim: {val_metrics['feature_similarity']:.4f}, "
                f"Domain Acc: {val_metrics['domain_accuracy']:.4f}")
    
    except KeyboardInterrupt:
        print("\nTraining wurde manuell unterbrochen. Speichere letztes Modell...")
        save_model(model, val_metrics, epoch, save_dir)
    
    print("\nTraining abgeschlossen!")
    wandb.finish()

if __name__ == "__main__":
    main()