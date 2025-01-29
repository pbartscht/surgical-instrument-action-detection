import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from ultralytics.nn.modules.conv import Concat

class InstrumentFeatureExtractor(nn.Module):
    def __init__(self, yolo_path):
        super().__init__()
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        self.feature_layer = 16
        
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.yolo_model.eval()
    
    def extract_features(self, x):
        """
        Extrahiert Features mit korrektem Handling verschiedener Feature Map Größen
        """
        feature_map = None
        intermediate_outputs = []
        
        print("\n=== Starting Feature Extraction ===")
        print(f"Initial input shape: {x.shape}")
        
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                print(f"\nProcessing Layer {i} ({type(layer).__name__}):")
                print(f"Input shape: {x.shape}")
                
                if isinstance(layer, nn.modules.upsampling.Upsample):
                    x = layer(x)
                    print(f"After Upsample: {x.shape}")
                    intermediate_outputs.append(x)
                    
                elif isinstance(layer, Concat):
                    print("Concat operation:")
                    tensors_to_cat = intermediate_outputs[-2:]
                    for idx, tensor in enumerate(tensors_to_cat):
                        print(f"  Tensor {idx} shape: {tensor.shape}")
                    
                    # Resize den zweiten Tensor auf die Größe des ersten
                    target_size = tensors_to_cat[0].shape[-2:]  # Nimm die räumlichen Dimensionen des ersten Tensors
                    tensors_to_cat[1] = nn.functional.interpolate(
                        tensors_to_cat[1],
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )
                    print(f"  After resize: {tensors_to_cat[1].shape}")
                    
                    x = torch.cat(tensors_to_cat, dim=1)
                    print(f"  After concat: {x.shape}")
                    intermediate_outputs = [x]
                    
                else:
                    x = layer(x)
                    intermediate_outputs.append(x)
                
                if i == self.feature_layer:
                    print(f"\nReached target layer {i}")
                    print(f"Feature map shape: {x.shape}")
                    feature_map = x.clone()
                    break
        
        if feature_map is None:
            raise ValueError(f"Layer {self.feature_layer} wurde nicht erreicht")
        
        batch_size, channels, height, width = feature_map.shape
        flattened_features = feature_map.view(batch_size, -1)
        print(f"\nFinal flattened feature shape: {flattened_features.shape}")
        
        return flattened_features
    
    def get_detections(self, x):
        """
        Führt nur die YOLO Detektionen durch
        """
        return self.yolo(x)

def extract_and_visualize_features(video_path, yolo_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    feature_extractor = InstrumentFeatureExtractor(yolo_path).to(device)
    
    # Zwei separate Transforms: einer für YOLO (resized) und einer für Feature Extraction
    yolo_transform = transforms.Compose([
        transforms.Resize((640, 640)),  # YOLO erwartet 640x640
        transforms.ToTensor(),
    ])
    
    feature_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    all_features = []
    all_labels = []
    frame_indices = []  # Speichert den Frame-Index für jedes Feature
    
    print("\nExtrahiere Features aus Video...")
    
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.png')])
    
    for frame_idx, frame_file in enumerate(tqdm(frame_files)):
        img_path = video_path / frame_file
        img = Image.open(img_path).convert('RGB')
        
        # Bild für YOLO vorbereiten
        yolo_input = yolo_transform(img).unsqueeze(0).to(device)
        
        # Bild für Feature Extraction vorbereiten
        feature_input = feature_transform(img).unsqueeze(0).to(device)
        
        # Erst YOLO Detektionen durchführen
        detections = feature_extractor.get_detections(yolo_input)
        
        # Dann Features extrahieren
        if len(detections[0].boxes) > 0:  # Wenn Instrumente erkannt wurden
            features = feature_extractor.extract_features(feature_input)
            features_np = features.cpu().numpy()
            
            # Für jedes erkannte Instrument im Frame den Feature-Vektor wiederholen
            frame_labels = [
                detections[0].names[int(box.cls)].lower() 
                for box in detections[0].boxes
            ]
            
            # Features für jedes Label im Frame wiederholen
            for _ in range(len(frame_labels)):
                all_features.append(features_np)
                frame_indices.append(frame_idx)
            
            all_labels.extend(frame_labels)
    
    if not all_features:
        print("Keine Features extrahiert! Überprüfen Sie, ob Instrumente erkannt wurden.")
        return None, None
        
    # Features zu Array zusammenfügen
    features_array = np.concatenate(all_features, axis=0)
    
    print(f"\nExtrahierte Features Shape: {features_array.shape}")
    print(f"Anzahl Labels: {len(all_labels)}")
    
    # Features und Labels speichern
    save_dir = Path('feature_spaces')
    save_dir.mkdir(exist_ok=True)
    
    np.savez(
        save_dir / 'raw_instrument_features_Cholect50_VID92_layer16.npz',
        features=features_array,
        labels=all_labels,
        frame_indices=frame_indices
    )
    
    # Visualisierung
    visualize_feature_space(features_array, all_labels)
    
    return features_array, all_labels

def visualize_feature_space(features, labels):
    print("\nBeginne Feature Space Visualisierung...")
    
    print("Führe t-SNE Dimensionsreduktion durch...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    
    unique_instruments = list(set(labels))
    colors = sns.color_palette('husl', n_colors=len(unique_instruments))
    
    for idx, instrument in enumerate(unique_instruments):
        mask = [label == instrument for label in labels]
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=instrument,
            alpha=0.6,
            s=50
        )
    
    plt.title('Raw Feature Space (Layer 16) nach Instrumententyp')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_dir = Path('feature_spaces')
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'raw_feature_space_16.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisierung gespeichert in: {save_dir}/raw_feature_space_16.png")
    
    print("\nFeature-Statistiken:")
    print(f"Feature Dimensionalität: {features.shape[1]}")
    print("Anzahl Features pro Instrument:")
    for instrument in unique_instruments:
        count = sum(1 for label in labels if label == instrument)
        print(f"- {instrument}: {count}")

if __name__ == "__main__":
    video_path = Path("/data/Bartscht/CholecT50/videos/VID92")
    yolo_path = Path("/data/Bartscht/YOLO/best_v35.pt")
    
    features, labels = extract_and_visualize_features(video_path, yolo_path)