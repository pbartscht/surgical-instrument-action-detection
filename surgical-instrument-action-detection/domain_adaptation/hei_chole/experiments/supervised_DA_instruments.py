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
    def __init__(self, yolo_path="/data/Bartscht/YOLO/best_v35.pt", num_classes=5):
        super().__init__()
        # YOLO als Feature Extractor - immer im eval mode
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        self.yolo_model.eval()
        self.feature_layer = 10
        
        # Rest der Komponenten wie gehabt
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.instrument_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def extract_features(self, x):
        features = None
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
                    break
        return self.feature_reducer(features)

    def forward(self, x, alpha=1.0):
        features = self.extract_features(x)
        
        # Domain Classification mit Gradient Reversal
        domain_pred = self.domain_classifier(
            GradientReversalLayer.apply(features, alpha)
        )
        
        # Instrument Classification
        instrument_pred = self.instrument_classifier(features)
        
        return domain_pred, instrument_pred

def train_epoch(model, dataloader, optimizer, device, epoch, domain_lambda=0.3):
    for param in model.parameters():
        param.requires_grad = True
    model.yolo_model.eval()  # Sicherstellen dass YOLO im eval bleibt
    
    total_loss = 0
    total_instrument_loss = 0
    total_domain_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['labels'].float().to(device)
        domains = batch['domain'].float().to(device)
        
        p = epoch / 100
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        optimizer.zero_grad()
        domain_pred, instrument_pred = model(images, alpha)
        
        instrument_loss = F.binary_cross_entropy(instrument_pred, labels)
        domain_loss = F.binary_cross_entropy(domain_pred.squeeze(), domains)
        loss = instrument_loss + domain_lambda * domain_loss
        
        loss.backward()
        optimizer.step()
        
        # Akkumuliere Verluste für Durchschnitt
        total_loss += loss.item()
        total_instrument_loss += instrument_loss.item()
        total_domain_loss += domain_loss.item()
        
        if batch_idx % 10 == 0:
            wandb.log({
                "batch_total_loss": loss.item(),
                "batch_instrument_loss": instrument_loss.item(),
                "batch_domain_loss": domain_loss.item(),
                "alpha": alpha
            })
            
            print(f'Epoch: {epoch} [{batch_idx}/{len(dataloader)}]')
            print(f'Total Loss: {loss.item():.4f}')
            print(f'Instrument Loss: {instrument_loss.item():.4f}')
            print(f'Domain Loss: {domain_loss.item():.4f}\n')
    
    # Durchschnittliche Verluste pro Epoch
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
    model.eval()
    total_val_loss = 0
    total_instrument_loss = 0
    total_domain_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            labels = batch['labels'].float().to(device)
            domains = batch['domain'].float().to(device)
            
            # Gleiche Alpha-Berechnung wie beim Training
            p = epoch / 100
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            domain_pred, instrument_pred = model(images, alpha)
            
            instrument_loss = F.binary_cross_entropy(instrument_pred, labels)
            domain_loss = F.binary_cross_entropy(domain_pred.squeeze(), domains)
            val_loss = instrument_loss + 0.3 * domain_loss
            
            total_val_loss += val_loss.item()
            total_instrument_loss += instrument_loss.item()
            total_domain_loss += domain_loss.item()
    
    # Durchschnittliche Verluste
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
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    state_dict = {
        'feature_reducer': model.feature_reducer.state_dict(),
        'instrument_classifier': model.instrument_classifier.state_dict()
    }
    torch.save(state_dict, save_dir / 'adapter_weights.pt')

def main():
    wandb.init(project="surgical-domain-adaptation")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train und Val Loader mit der gleichen Funktion
    train_loader = balanced_dataloader(split='train')
    val_loader = balanced_dataloader(split='val')
    
    model = DomainAdapter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    save_dir = Path("adapter_checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate_epoch(model, val_loader, device, epoch)
        
        # Model Saving & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_adapter(model, save_dir / "best_adapter")
            wandb.log({"best_val_loss": val_loss})
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered nach Epoch {epoch}")
            break
            
        if epoch % 5 == 0:
            save_adapter(model, save_dir / f"adapter_epoch_{epoch}")
    
    wandb.finish()

if __name__ == "__main__":
    main()