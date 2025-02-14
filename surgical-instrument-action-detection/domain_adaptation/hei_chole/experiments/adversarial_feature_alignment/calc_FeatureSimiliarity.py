from dataloader import balanced_dataloader
import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
from pathlib import Path

class FeatureExtractor(nn.Module):
    def __init__(self, yolo_path="/data/Bartscht/YOLO/best_v35.pt"):
        super().__init__()
        # YOLO im eval mode
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        self.feature_layer = 8
        
        # YOLO Training deaktivieren
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.yolo_model.eval()
        
        # Exakt gleiche Feature Reducer Architektur wie im Training
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5)  # Gleiches Dropout wie im Training
        )
        
        # Lade trainierte Gewichte
        checkpoint = torch.load('/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/domain_adapter_weights/best_feature_reducer.pt')
        self.feature_reducer.load_state_dict(checkpoint['feature_reducer'])
        
    def set_train_mode(self, mode=False):
        self.feature_reducer.train(mode)
        return self
        
    def forward(self, x):
        # Feature Extraction
        features = None
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
                    break
        
        # Feature Reduction
        reduced_features = self.feature_reducer(features)
        return reduced_features

def evaluate_features(model, test_loader, device):
    """Evaluiere die Feature Qualität"""
    model.set_train_mode(False)
    feature_similarities = []
    source_features_all = []
    target_features_all = []
    
    print("\nExtrahiere Features...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            domains = batch['domain'].to(device)
            
            # Features extrahieren
            features = model(images)
            
            # Features nach Domänen trennen
            source_mask = domains == 0
            target_mask = domains == 1
            
            if source_mask.any():
                source_features_all.append(features[source_mask])
            if target_mask.any():
                target_features_all.append(features[target_mask])
    
    # Konvertiere Listen zu Tensoren
    if source_features_all and target_features_all:
        source_features = torch.cat(source_features_all, dim=0)
        target_features = torch.cat(target_features_all, dim=0)
        
        # Berechne durchschnittliche Source Features
        mean_source = source_features.mean(0, keepdim=True)
        
        # Berechne Ähnlichkeit für jedes Target Feature
        cosine_sim = nn.CosineSimilarity(dim=1)
        similarities = cosine_sim(
            mean_source.expand(target_features.size(0), -1),
            target_features
        )
        
        # Statistiken
        avg_sim = similarities.mean().item()
        min_sim = similarities.min().item()
        max_sim = similarities.max().item()
        std_sim = similarities.std().item()
        
        print(f"\nFeature Ähnlichkeits-Statistiken:")
        print(f"Durchschnittliche Ähnlichkeit: {avg_sim:.4f}")
        print(f"Minimale Ähnlichkeit: {min_sim:.4f}")
        print(f"Maximale Ähnlichkeit: {max_sim:.4f}")
        print(f"Standardabweichung: {std_sim:.4f}")
        
        # Berechne Prozentsatz der Features mit hoher Ähnlichkeit
        high_sim_threshold = 0.7
        high_sim_ratio = (similarities > high_sim_threshold).float().mean().item()
        print(f"Anteil Features mit Ähnlichkeit > {high_sim_threshold}: {high_sim_ratio*100:.1f}%")
        
        return {
            'avg_similarity': avg_sim,
            'min_similarity': min_sim,
            'max_similarity': max_sim,
            'std_similarity': std_sim,
            'high_similarity_ratio': high_sim_ratio
        }
    else:
        print("Keine ausreichenden Daten für Vergleich gefunden!")
        return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model initialisieren
    model = FeatureExtractor().to(device)
    
    # Test Dataloader
    test_loader = balanced_dataloader(split='test')
    
    # Feature Evaluation
    metrics = evaluate_features(model, test_loader, device)
    
    if metrics:
        # Speichere Ergebnisse
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "feature_evaluation.txt", "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()