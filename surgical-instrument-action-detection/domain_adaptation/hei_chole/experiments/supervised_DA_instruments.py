import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from dataset import SurgicalDataset, TOOL_MAPPING, CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING, HEICHOLE_CLASSES
from ultralytics import YOLO
import matplotlib as plt

def load_yolo_model(yolo_path, device):
    """Load and prepare YOLO model for feature extraction"""
    model = YOLO(yolo_path)
    model.to(device)
    model.model.eval()  # Set to evaluation mode
    return model

class DomainAdaptationModel(nn.Module):
    def __init__(self, yolo_path, device, feature_layer=10):
        super().__init__()
        self.device = device
        self.feature_layer = feature_layer
        
        # Load and freeze YOLO model
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(device)
        self.yolo_model.model.eval()
        
        # Feature Reducer
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Classifier für CholecT50 Format (6 Klassen)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, len(TOOL_MAPPING)),
            nn.Sigmoid()
        )
        
        # Erstelle Mapping Matrix beim Initialisieren
        self.register_buffer('mapping_matrix', self.create_mapping_matrix())
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def create_mapping_matrix(self):
        """Erstelle die Mapping Matrix zwischen CholecT50 und HeiChole Klassen"""
        source_classes = len(TOOL_MAPPING)  # 6 Klassen
        target_classes = len(HEICHOLE_CLASSES)  # 7 Klassen
        mapping_matrix = torch.zeros(source_classes, target_classes)
        
        # Explizites Mapping
        for source_idx, source_tool in TOOL_MAPPING.items():
            target_tool = CHOLECT50_TO_HEICHOLE_INSTRUMENT_MAPPING.get(source_tool)
            if target_tool:
                target_idx = [k for k, v in HEICHOLE_CLASSES.items() if v == target_tool][0]
                mapping_matrix[source_idx, target_idx] = 1
        
        return mapping_matrix
    
    def extract_features(self, images):
        """Extract features from specific YOLO layer with dimension validation"""
        images = images.to(self.device)
        x = images
        features = None
        
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model.model.model):
                layer = layer.to(self.device)
                x = layer(x)
                if i == self.feature_layer:
                    features = x
                    # Validiere Feature Dimensionen
                    self.validate_feature_dimensions(features)
                    break
        
        return features
    
    def validate_feature_dimensions(self, features):
        """Überprüfe ob Features die erwartete Dimension haben"""
        B, C, H, W = features.shape
        assert C == 512, f"Expected 512 channels, got {C}"
        logging.info(f"Feature dimensions: Batch={B}, Channels={C}, Height={H}, Width={W}")
    
    def debug_feature_maps(self, features, save_dir='feature_maps'):
        """Visualisiere und speichere Feature Maps zur Analyse"""
        B, C, H, W = features.shape
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Speichere Statistiken
        feature_stats = {
            'mean': features.mean().item(),
            'std': features.std().item(),
            'min': features.min().item(),
            'max': features.max().item(),
            'shape': features.shape
        }
        
        logging.info(f"Feature Statistics: {feature_stats}")
        
        # Visualisiere einige Feature Maps
        for i in range(min(5, C)):  # Erste 5 Kanäle als Beispiel
            feature_map = features[0, i].cpu().numpy()  # Erstes Batch-Element
            plt.figure(figsize=(5,5))
            plt.imshow(feature_map, cmap='viridis')
            plt.colorbar()
            plt.title(f'Feature Map {i}')
            plt.savefig(save_dir / f'feature_map_{i}.png')
            plt.close()
    
    def forward(self, x):
        # Feature Extraction from YOLO
        yolo_features = self.extract_features(x)
        
        # Feature Reduction
        reduced_features = self.feature_reducer(yolo_features)
        
        # Classification im CholecT50 Format
        cholect_pred = self.classifier(reduced_features)
        
        # Mapping auf HeiChole Format
        heichole_pred = self.map_to_heichole(cholect_pred)
        
        return cholect_pred, heichole_pred
    
    def map_to_heichole(self, cholect_pred):
        """Mappe CholecT50 Vorhersagen auf HeiChole Format"""
        # Anwendung des Mappings
        heichole_pred = torch.matmul(cholect_pred, self.mapping_matrix)
        return torch.clamp(heichole_pred, 0, 1)  # Ensure valid probabilities
    
    def freeze_yolo(self):
        """Friere YOLO Parameter ein"""
        for param in self.yolo_model.parameters():
            param.requires_grad = False
    
    def log_model_info(self):
        """Logge Model Informationen für Debugging"""
        logging.info("Model Architecture:")
        logging.info(f"YOLO Feature Layer: {self.feature_layer}")
        logging.info(f"Mapping Matrix Shape: {self.mapping_matrix.shape}")
        logging.info("Feature Reducer Architecture:")
        logging.info(self.feature_reducer)
        logging.info("Classifier Architecture:")
        logging.info(self.classifier)

class SupervisedDomainAdapter:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        self.scheduler = torch.optim.ReduceLROnPlateau(
            self.optimizer, mode='min', 
            patience=5, factor=0.5
        )
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Daten auf Device verschieben
        images = batch['image'].to(self.device)
        labels = batch['labels'].to(self.device)
        domain = batch['domain'].to(self.device)
        
        # Forward pass
        cholect_pred, heichole_pred = self.model(images)
        
        # Loss Berechnung für beide Domains
        loss = 0
        metrics = {}
        
        # Source Domain (CholecT50) Loss
        source_mask = (domain == 0)
        if source_mask.any():
            source_loss = F.binary_cross_entropy(
                cholect_pred[source_mask],
                labels[source_mask]
            )
            loss += source_loss
            metrics['source_loss'] = source_loss.item()
        
        # Target Domain (HeiChole) Loss
        target_mask = (domain == 1)
        if target_mask.any():
            target_loss = F.binary_cross_entropy(
                heichole_pred[target_mask],
                labels[target_mask]
            )
            loss += target_loss
            metrics['target_loss'] = target_loss.item()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        metrics['total_loss'] = loss.item()
        return metrics
    
    def validate(self, val_loader):
        self.model.eval()
        metrics = {
            'val_loss': 0,
            'source_correct': 0,
            'source_total': 0,
            'target_correct': 0,
            'target_total': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                domain = batch['domain'].to(self.device)
                
                cholect_pred, heichole_pred = self.model(images)
                
                # Separate metrics for each domain
                source_mask = (domain == 0)
                target_mask = (domain == 1)
                
                if source_mask.any():
                    source_loss = F.binary_cross_entropy(
                        cholect_pred[source_mask],
                        labels[source_mask]
                    )
                    metrics['val_loss'] += source_loss.item()
                    
                    # Binary accuracy for source domain
                    source_correct = ((cholect_pred[source_mask] > 0.5) == 
                                   (labels[source_mask] > 0.5)).float().sum()
                    metrics['source_correct'] += source_correct.item()
                    metrics['source_total'] += source_mask.sum().item() * len(TOOL_MAPPING)
                
                if target_mask.any():
                    target_loss = F.binary_cross_entropy(
                        heichole_pred[target_mask],
                        labels[target_mask]
                    )
                    metrics['val_loss'] += target_loss.item()
                    
                    # Binary accuracy for target domain
                    target_correct = ((heichole_pred[target_mask] > 0.5) == 
                                   (labels[target_mask] > 0.5)).float().sum()
                    metrics['target_correct'] += target_correct.item()
                    metrics['target_total'] += target_mask.sum().item() * len(TOOL_MAPPING)
        
        # Calculate final metrics
        metrics['val_loss'] /= len(val_loader)
        if metrics['source_total'] > 0:
            metrics['source_accuracy'] = metrics['source_correct'] / metrics['source_total']
        if metrics['target_total'] > 0:
            metrics['target_accuracy'] = metrics['target_correct'] / metrics['target_total']
        
        return metrics

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize datasets
    source_dataset = SurgicalDataset(
        dataset_dir="path/to/cholect50",
        dataset_type='source'
    )
    target_dataset = SurgicalDataset(
        dataset_dir="path/to/heichole",
        dataset_type='target'
    )
    
    # Create data loaders
    source_loader = DataLoader(
        source_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = load_yolo_model()  # Implementation needed
    model = DomainAdaptationModel(yolo_model).to(device)
    adapter = SupervisedDomainAdapter(model, device)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_metrics = {'epoch': epoch}
        
        # Training
        for batch_idx, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
            combined_batch = {
                'image': torch.cat([source_batch['image'], target_batch['image']]),
                'labels': torch.cat([source_batch['labels'], target_batch['labels']]),
                'domain': torch.cat([source_batch['domain'], target_batch['domain']])
            }
            
            metrics = adapter.train_step(combined_batch)
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: {metrics}")
        
        # Validation
        val_metrics = adapter.validate(target_loader)
        epoch_metrics.update(val_metrics)
        
        # Log metrics
        logger.info(f"Epoch {epoch} completed: {epoch_metrics}")
        
        # Learning rate scheduling
        adapter.scheduler.step(val_metrics['val_loss'])

if __name__ == "__main__":
    main()