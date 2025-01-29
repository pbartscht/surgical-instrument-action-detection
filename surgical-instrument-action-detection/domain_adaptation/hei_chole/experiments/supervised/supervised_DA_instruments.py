from dataloader import balanced_dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import wandb
from pathlib import Path

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainAdapter(nn.Module):
    def __init__(self, yolo_path="/data/Bartscht/YOLO/best_v35.pt", yolo_classes=6, target_classes=5):
        super().__init__()
        # YOLO Initialisierung bleibt gleich
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.yolo_model.eval()
        self.feature_layer = 10
        
        # Mapping Matrix bleibt gleich
        self.register_buffer('mapping_matrix', torch.zeros(yolo_classes, target_classes))
        self.mapping_matrix[0, 0] = 1  # grasper -> grasper
        self.mapping_matrix[1, 2] = 1  # bipolar -> coagulation
        self.mapping_matrix[2, 2] = 1  # hook -> coagulation
        self.mapping_matrix[3, 3] = 1  # scissors -> scissors
        self.mapping_matrix[4, 1] = 1  # clipper -> clipper
        self.mapping_matrix[5, 4] = 1  # irrigator -> suction_irrigation
        
        # Vereinfachter Feature Reducer
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)  # Erhöhter Dropout
        )

        # Vereinfachter Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Vereinfachter Instrument Classifier
        self.instrument_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, yolo_classes),
            nn.Sigmoid()
        )

    def extract_features(self, x):
        self.yolo_model.eval()
        features = None
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
                    break
        return self.feature_reducer(features)

    def forward_domain_classifier(self, x, alpha):
        for i, layer in enumerate(self.domain_classifier):
            if i == 1:  # Residual Block
                identity = x
                x = layer(x) + identity
            elif i == 2:  # Attention Block
                attention = layer(x)
                x = x * attention
            else:
                x = layer(x)
        return x

    def forward_instrument_classifier(self, x):
        for i, layer in enumerate(self.instrument_classifier):
            if i == 4:  # Context Integration Block
                identity = x
                x = layer(x) + identity
            else:
                x = layer(x)
        return x

    def forward(self, x, alpha=1.0):
        features = self.extract_features(x)
        
        # Domain Classification mit Gradient Reversal und verbessertem Forward
        domain_features = GradientReversalLayer.apply(features, alpha)
        domain_pred = self.forward_domain_classifier(domain_features, alpha)
        
        # Instrument Classification mit verbessertem Forward
        yolo_pred = self.forward_instrument_classifier(features)
        
        # Mapping auf HeiChole Klassen
        heichole_pred = torch.matmul(yolo_pred, self.mapping_matrix)
        
        return domain_pred, yolo_pred, heichole_pred

    def train(self, mode=True):
        super().train(mode)
        self.yolo_model.eval()
        return self

    def eval(self):
        super().eval()
        self.yolo_model.eval()
        return self
    
def get_alpha(epoch, num_epochs=30):
    """Implementierung eines Warm-up Schedules für GRL"""
    p = epoch / num_epochs
    return 2. / (1. + np.exp(-10 * p)) - 1

def train_epoch(model, dataloader, optimizer, device, epoch, domain_lambda=0.3):
    model.feature_reducer.train()
    model.domain_classifier.train()
    model.instrument_classifier.train()
    
    total_loss = 0
    total_instrument_loss = 0
    total_domain_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['labels'].float().clamp(0, 1).to(device)
        domains = batch['domain'].float().clamp(0, 1).to(device)
        
        alpha = get_alpha(epoch)
        optimizer.zero_grad()
        
        # Forward Pass mit neuer Struktur
        domain_pred, yolo_pred, heichole_pred = model(images, alpha)
        
        # Losses berechnen
        instrument_loss = F.binary_cross_entropy(heichole_pred.clamp(1e-7, 1), labels)
        domain_loss = F.binary_cross_entropy(domain_pred.squeeze().clamp(1e-7, 1), domains)
        
        # Gesamtverlust
        loss = instrument_loss + domain_lambda * domain_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_instrument_loss += instrument_loss.item()
        total_domain_loss += domain_loss.item()
        
        if batch_idx % 10 == 0:
            wandb.log({
                "batch_total_loss": loss.item(),
                "batch_instrument_loss": instrument_loss.item(),
                "batch_domain_loss": domain_loss.item(),
                "alpha": alpha,
                "yolo_pred_mean": yolo_pred.mean().item(),  # Zusätzliche Metriken
                "heichole_pred_mean": heichole_pred.mean().item()
            })
            
            print(f'Epoch: {epoch} [{batch_idx}/{len(dataloader)}]')
            print(f'Total Loss: {loss.item():.4f}')
            print(f'Instrument Loss: {instrument_loss.item():.4f}')
            print(f'Domain Loss: {domain_loss.item():.4f}\n')
    
    avg_loss = total_loss / len(dataloader)
    avg_instrument_loss = total_instrument_loss / len(dataloader)
    avg_domain_loss = total_domain_loss / len(dataloader)
    
    wandb.log({
        "epoch": epoch,
        "avg_total_loss": avg_loss,
        "avg_instrument_loss": avg_instrument_loss,
        "avg_domain_loss": avg_domain_loss
    })
    
    return avg_loss

def validate_epoch(model, val_loader, device, epoch):
    model.feature_reducer.eval()
    model.domain_classifier.eval()
    model.instrument_classifier.eval()
    
    total_val_loss = 0
    total_instrument_loss = 0
    total_domain_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            labels = batch['labels'].float().clamp(0, 1).to(device)
            domains = batch['domain'].float().clamp(0, 1).to(device)
            
            alpha = get_alpha(epoch)
            
            # Forward Pass mit neuer Struktur
            domain_pred, yolo_pred, heichole_pred = model(images, alpha)
            
            # Losses berechnen
            instrument_loss = F.binary_cross_entropy(heichole_pred.clamp(1e-7, 1), labels)
            domain_loss = F.binary_cross_entropy(domain_pred.squeeze().clamp(1e-7, 1), domains)
            val_loss = instrument_loss + 0.3 * domain_loss
            
            total_val_loss += val_loss.item()
            total_instrument_loss += instrument_loss.item()
            total_domain_loss += domain_loss.item()
            
            if batch_idx % 10 == 0:  # Zusätzliches Logging während der Validierung
                wandb.log({
                    "val_batch_loss": val_loss.item(),
                    "val_yolo_pred_mean": yolo_pred.mean().item(),
                    "val_heichole_pred_mean": heichole_pred.mean().item()
                })
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_instrument_loss = total_instrument_loss / len(val_loader)
    avg_domain_loss = total_domain_loss / len(val_loader)
    
    wandb.log({
        "val_total_loss": avg_val_loss,
        "val_instrument_loss": avg_instrument_loss,
        "val_domain_loss": avg_domain_loss,
        "val_epoch": epoch
    })
    
    return avg_val_loss

def save_adapter(model, save_dir):
    try:
        save_dir = Path(save_dir)
        print(f"Versuche Verzeichnis zu erstellen: {save_dir.absolute()}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if not save_dir.exists():
            raise RuntimeError(f"Konnte Verzeichnis nicht erstellen: {save_dir}")
        
        # Erweitere state dict um Domain Classifier
        state_dict = {
            'feature_reducer': model.feature_reducer.state_dict(),
            'instrument_classifier': model.instrument_classifier.state_dict(),
            'domain_classifier': model.domain_classifier.state_dict()  # Neu hinzugefügt
        }
        
        save_path = save_dir / 'adapter_weights.pt'
        print(f"Versuche zu speichern unter: {save_path.absolute()}")
        
        torch.save(state_dict, save_path)
        
        if save_path.exists():
            print(f"Erfolgreich gespeichert! Dateigröße: {save_path.stat().st_size / 1024:.2f} KB")
        else:
            raise RuntimeError(f"Datei wurde nicht erstellt: {save_path}")
            
    except Exception as e:
        print(f"Fehler beim Speichern: {str(e)}")
        raise

def main():
    # Wandb Konfiguration
    config = {
        "learning_rate": 0.001,
        "num_epochs": 30,
        "batch_size": 32,
        "domain_lambda": 0.3,
        "patience": 5,
        "yolo_classes": 6,
        "target_classes": 5,
        "model_path": "/data/Bartscht/YOLO/best_v35.pt"
    }
    
    wandb.init(
        project="surgical-domain-adaptation",
        config=config
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataloader
    train_loader = balanced_dataloader(split='train')
    val_loader = balanced_dataloader(split='val')
    
    # Model Initialization
    model = DomainAdapter(
        yolo_path=config["model_path"],
        yolo_classes=config["yolo_classes"],
        target_classes=config["target_classes"]
    ).to(device)
    
    # Optimizer mit Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Training Setup
    best_val_loss = float('inf')
    patience_counter = 0
    save_dir = Path("adapter_weights")
    print(f"Base save directory: {save_dir.absolute()}")
    
    # Versuch die Checkpoints zu laden falls vorhanden
    checkpoint_path = save_dir / 'latest_checkpoint.pt'
    start_epoch = 0
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Training Loop
    for epoch in range(start_epoch, config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            epoch, 
            domain_lambda=config["domain_lambda"]
        )
        
        # Validation
        val_loss = validate_epoch(model, val_loader, device, epoch)
        
        # Learning Rate Scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr})
        
        # Model Saving & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Speichere bestes Modell
            save_adapter(model, save_dir)
            wandb.log({
                "best_val_loss": val_loss,
                "best_model_epoch": epoch
            })
            
            # Speichere Checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, save_dir / 'latest_checkpoint.pt')
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= config["patience"]:
            print(f"Early stopping triggered nach Epoch {epoch}")
            wandb.log({"early_stopping_epoch": epoch})
            break
        
        # Periodisches Speichern
        if epoch % 5 == 0:
            save_adapter(model, save_dir / f"adapter_epoch_{epoch}")
            
        # Log Training Progress
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "patience_counter": patience_counter
        })
    
    # Final Logging und Cleanup
    wandb.log({
        "final_val_loss": val_loss,
        "best_val_loss_overall": best_val_loss,
        "total_epochs_trained": epoch + 1
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()