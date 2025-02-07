import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
from dataloader import balanced_dataloader

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

class ClassAwareSpatialAdapter(nn.Module):
    def __init__(self, yolo_path="/data/Bartscht/YOLO/best_v35.pt"):
        super().__init__()
        # YOLO Setup
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        self.feature_layer = 9
        
        # Freeze YOLO
        self.yolo_model.eval()
        for param in self.yolo_model.parameters():
            param.requires_grad = False
            
        # Feature Reducer
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=32),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Class Predictor
        self.class_predictor = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 5, 1)  # 6 instruments, but hook and bipolar mapped to  "coagulation"
        )
        self.current_step = 0
        self.warmup_steps = 1000

    def set_train_mode(self, mode=True):
        self.feature_reducer.train(mode)
        self.domain_classifier.train(mode)
        self.class_predictor.train(mode)
        self.yolo_model.eval()
        return self

    def forward(self, images, domains, alpha=None, return_features=False):
        
        if alpha is None:
            alpha = min(1.0, self.current_step / self.warmup_steps)
            self.current_step += 1

        # Feature Extraction
        with torch.no_grad():
            x = images.clone()
            features = None
            for i, layer in enumerate(self.yolo_model):
                if i > self.feature_layer:
                    break
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
        
        # Feature Reduction
        reduced_features = self.feature_reducer(features)
        # Domain Classification mit GRL
        domain_features = GradientReversalLayer.apply(reduced_features, alpha)
        domain_pred = self.domain_classifier(domain_features)
        
        # Class Prediction
        class_pred = self.class_predictor(reduced_features)
        
        if return_features:
            return {
                'domain_pred': domain_pred,
                'class_pred': class_pred,
                'features': reduced_features
            }
        return domain_pred

def spatial_consistency_loss(source_features, target_features):
    """Berechnet Feature-Consistency über räumliche Dimensionen"""
    mean_source = source_features.mean(dim=0, keepdim=True)
    similarity = F.cosine_similarity(
        mean_source.expand(target_features.size(0), -1, -1, -1),
        target_features,
        dim=1
    ).mean()
    return 1 - similarity

def class_aware_consistency_loss(source_features, target_features, source_labels):
    """Berechnet Feature-Consistency pro Klasse"""
    total_similarity = 0
    num_classes = source_labels.size(1)
    weights = []
    
    for class_idx in range(num_classes):
        class_mask = source_labels[:, class_idx] > 0.5
        if class_mask.any():
            class_source_features = source_features[class_mask]
            # Stärkere Gewichtung für seltenere Klassen
            weight = 1.0 / (class_mask.float().mean() + 1e-6)
            weights.append(weight)
            
            similarity = spatial_consistency_loss(class_source_features, target_features)
            total_similarity += weight * similarity
            
    return total_similarity / sum(weights) if weights else torch.tensor(0.0).to(source_features.device)

def calculate_spatial_metrics(domain_preds, domains):
    """Berechnet Metriken basierend auf den Domain-Predictions"""
    B, _, H_feat, W_feat = domain_preds.shape
    domains_expanded = domains.view(B, 1, 1, 1).expand(B, 1, H_feat, W_feat).float()
    
    preds_binary = (domain_preds > 0.5).float()
    accuracy = (preds_binary.eq(domains_expanded)).float().mean()
    
    source_mask = (domains_expanded == 0)
    target_mask = (domains_expanded == 1)
    
    source_acc = (preds_binary[source_mask].eq(domains_expanded[source_mask])).float().mean() if source_mask.sum() > 0 else torch.tensor(0.0)
    target_acc = (preds_binary[target_mask].eq(domains_expanded[target_mask])).float().mean() if target_mask.sum() > 0 else torch.tensor(0.0)
    
    domain_confusion = 1 - torch.abs(2 * accuracy - 1)
    
    return {
        'accuracy': accuracy.item(),
        'source_acc': source_acc.item(),
        'target_acc': target_acc.item(),
        'domain_confusion': domain_confusion.item()
    }

def train_epoch(model, dataloader, optimizer_reducer, optimizer_classifier, device, epoch, config):
    model.set_train_mode(True)
    metrics_tracker = {
        'critic_loss': [],
        'artist_loss': [],
        'class_loss': [],
        'feature_sim': [],
        'domain_metrics': []
    }
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch+1}/{config["num_epochs"]}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        domains = batch['domain'].to(device)
        labels = batch['labels'].to(device)

        # 1. Training des Domain Classifiers
        optimizer_classifier.zero_grad()
        outputs = model(images, domains, alpha=1.0, return_features=True)
        domain_pred = outputs['domain_pred']
        
        B, _, H_feat, W_feat = domain_pred.shape
        expanded_domains = domains.view(B, 1, 1, 1).expand(B, 1, H_feat, W_feat).float()
        
        critic_loss = F.binary_cross_entropy(domain_pred, expanded_domains)
        critic_loss.backward()
        optimizer_classifier.step()

        # 2. Training des Feature Reducers (mit Class Predictor)
        optimizer_reducer.zero_grad()
        outputs = model(images, domains, alpha=1.0, return_features=True)
        domain_pred = outputs['domain_pred']
        class_pred = outputs['class_pred']
        reduced_features = outputs['features']
        
        # Domain Confusion Loss
        domain_confusion_loss = F.binary_cross_entropy(domain_pred, expanded_domains)
        
        # Class Loss (nur für Source Domain)
        source_mask = domains == 0
        if source_mask.any():
            class_pred_source = class_pred[source_mask].mean(dim=(2, 3))
            class_loss = F.binary_cross_entropy_with_logits(
                class_pred_source,
                labels[source_mask]
            )
        else:
            class_loss = torch.tensor(0.0).to(device)
        
        # Feature Consistency (klassenweise)
        source_features = reduced_features[domains == 0]
        target_features = reduced_features[domains == 1]
        source_labels = labels[domains == 0]
        
        if source_features.size(0) > 0 and target_features.size(0) > 0:
            consistency_loss = class_aware_consistency_loss(
                source_features, target_features, source_labels
            )
        else:
            consistency_loss = torch.tensor(0.0).to(device)
        
        # Kombinierter Loss für Feature Reducer
        total_loss = (
            domain_confusion_loss +
            config['class_weight'] * class_loss +
            config['feature_consistency_weight'] * consistency_loss
        )
        
        total_loss.backward()
        optimizer_reducer.step()
        
        # Metrics Tracking
        metrics_tracker['critic_loss'].append(critic_loss.item())
        metrics_tracker['artist_loss'].append(-domain_confusion_loss.item())
        metrics_tracker['class_loss'].append(class_loss.item())
        metrics_tracker['feature_sim'].append(consistency_loss.item() if consistency_loss > 0 else 0)
        metrics_tracker['domain_metrics'].append(
            calculate_spatial_metrics(domain_pred.detach(), domains)
        )
        
        if batch_idx % 10 == 0:
            critic_avg = np.mean(metrics_tracker['critic_loss'][-50:])
            artist_avg = np.mean(metrics_tracker['artist_loss'][-50:])
            class_avg = np.mean(metrics_tracker['class_loss'][-50:])
            pbar.set_description(
                f'Epoch {epoch+1} - Critic: {critic_avg:.4f}, Artist: {artist_avg:.4f}, Class: {class_avg:.4f}'
            )
    
    return {
        'critic_loss': np.mean(metrics_tracker['critic_loss']),
        'artist_loss': np.mean(metrics_tracker['artist_loss']),
        'class_loss': np.mean(metrics_tracker['class_loss']),
        'feature_similarity': np.mean(metrics_tracker['feature_sim']),
        'domain_accuracy': np.mean([m['accuracy'] for m in metrics_tracker['domain_metrics']]),
        'domain_confusion': np.mean([m['domain_confusion'] for m in metrics_tracker['domain_metrics']])
    }

def validate_epoch(model, dataloader, device, epoch, config):
    model.set_train_mode(False)
    metrics_tracker = {
        'critic_loss': [],
        'artist_loss': [],
        'class_loss': [],
        'feature_sim': [],
        'domain_metrics': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Val Epoch {epoch+1}'):
            images = batch['image'].to(device)
            domains = batch['domain'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images, domains, alpha=0.2, return_features=True)
            domain_pred = outputs['domain_pred']
            class_pred = outputs['class_pred']
            reduced_features = outputs['features']
            
            B, _, H_feat, W_feat = domain_pred.shape
            expanded_domains = domains.view(B, 1, 1, 1).expand(B, 1, H_feat, W_feat).float()
            
            critic_loss = F.binary_cross_entropy(domain_pred, expanded_domains)
            
            # Class Loss (nur für Source Domain)
            source_mask = domains == 0
            if source_mask.any():
                class_pred_source = class_pred[source_mask].mean(dim=(2, 3))
                class_loss = F.binary_cross_entropy_with_logits(
                    class_pred_source,
                    labels[source_mask]
                )
            else:
                class_loss = torch.tensor(0.0).to(device)
            
            # Feature Consistency
            source_features = reduced_features[domains == 0]
            target_features = reduced_features[domains == 1]
            source_labels = labels[domains == 0]
            
            if source_features.size(0) > 0 and target_features.size(0) > 0:
                consistency_loss = class_aware_consistency_loss(
                    source_features, target_features, source_labels
                )
            else:
                consistency_loss = torch.tensor(0.0).to(device)
            
            # Metrics Tracking
            metrics_tracker['critic_loss'].append(critic_loss.item())
            metrics_tracker['class_loss'].append(class_loss.item())
            metrics_tracker['feature_sim'].append(consistency_loss.item() if consistency_loss > 0 else 0)
            metrics_tracker['domain_metrics'].append(
                calculate_spatial_metrics(domain_pred, domains)
            )
    
    return {
        'critic_loss': np.mean(metrics_tracker['critic_loss']),
        'class_loss': np.mean(metrics_tracker['class_loss']),
        'feature_similarity': np.mean(metrics_tracker['feature_sim']),
        'domain_accuracy': np.mean([m['accuracy'] for m in metrics_tracker['domain_metrics']]),
        'domain_confusion': np.mean([m['domain_confusion'] for m in metrics_tracker['domain_metrics']])
    }
def save_model(model, metrics, epoch, save_dir):
    """Speichert Model-Checkpoints mit relevanten Metriken"""
    checkpoint = {
        'feature_reducer': model.feature_reducer.state_dict(),
        'domain_classifier': model.domain_classifier.state_dict(),
        'class_predictor': model.class_predictor.state_dict(),
        'metrics': metrics,
        'epoch': epoch,
        'spatial_info': {
            'input_shape': [512, None, None],  # H/16, W/16 werden dynamisch angepasst
            'output_shape': [512, None, None]  # H/16, W/16 werden dynamisch angepasst
        }
    }
    
    # Speichere komplettes Checkpoint
    torch.save(checkpoint, save_dir / f'class_aware_model_epoch_{epoch}.pt')
    
    # Speichere nur Feature Reducer für einfache Weiterverwendung
    torch.save({
        'state_dict': model.feature_reducer.state_dict(),
        'spatial_info': checkpoint['spatial_info']
    }, save_dir / 'class_aware_feature_reducer.pt')

def main():
    config = {
        "learning_rate": 0.0001,
        "num_epochs": 30,
        "batch_size": 32,
        "patience": 10,
        "model_path": "/data/Bartscht/YOLO/best_v35.pt",
        "feature_consistency_weight": 0.5, # # !! von 0.1 - stärkerer Fokus auf Feature-Erhaltung
        "class_weight": 1.0,        #Erhöht!!
        "min_lr": 1e-6,
        "experiment_name": "class_aware_domain_adaptation"
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
    
    # Model Setup mit class-aware Adapter
    model = ClassAwareSpatialAdapter(yolo_path=config["model_path"]).to(device)
    
    # Optimierer Setup
    optimizer_reducer = torch.optim.AdamW(
        list(model.feature_reducer.parameters()) + 
        list(model.class_predictor.parameters()),  # Feature Reducer + Class Predictor
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
    best_scores = {
        'feature_sim': -float('inf'),
        'class_loss': float('inf')
    }
    patience_counter = 0
    save_dir = Path(f"class_aware_adapter_weights_{config['experiment_name']}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting Class-Aware Domain Adaptation Training...")
    
    try:
        for epoch in range(config["num_epochs"]):
            # Training
            train_metrics = train_epoch(
                model, 
                train_loader, 
                optimizer_reducer, 
                optimizer_classifier, 
                device, 
                epoch, 
                config
            )
            
            # Validation
            val_metrics = validate_epoch(
                model, 
                val_loader, 
                device, 
                epoch, 
                config
            )
            
            # Scheduler Steps
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
            
            # Model Saving basierend auf Feature-Similarity UND Class Loss
            score = val_metrics['feature_similarity'] - config['class_weight'] * val_metrics['class_loss']
            if score > best_scores['feature_sim'] - config['class_weight'] * best_scores['class_loss']:
                best_scores['feature_sim'] = val_metrics['feature_similarity']
                best_scores['class_loss'] = val_metrics['class_loss']
                save_model(model, val_metrics, epoch, save_dir)
                patience_counter = 0
                print(f"\nNeues bestes Modell gespeichert (Score: {score:.4f})")
            else:
                patience_counter += 1
            
            # Early Stopping
            if patience_counter >= config["patience"]:
                print(f"\nEarly Stopping nach Epoch {epoch+1}")
                break
            
            # Epoch Summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Train - Critic Loss: {train_metrics['critic_loss']:.4f}, "
                  f"Class Loss: {train_metrics['class_loss']:.4f}, "
                  f"Feature Sim: {train_metrics['feature_similarity']:.4f}, "
                  f"Domain Acc: {train_metrics['domain_accuracy']:.4f}")
            print(f"Val - Critic Loss: {val_metrics['critic_loss']:.4f}, "
                  f"Class Loss: {val_metrics['class_loss']:.4f}, "
                  f"Feature Sim: {val_metrics['feature_similarity']:.4f}, "
                  f"Domain Acc: {val_metrics['domain_accuracy']:.4f}")
    
    except KeyboardInterrupt:
        print("\nTraining wurde manuell unterbrochen. Speichere letztes Modell...")
        save_model(model, val_metrics, epoch, save_dir)
    
    print("\nTraining abgeschlossen!")
    wandb.finish()

if __name__ == "__main__":
    main()