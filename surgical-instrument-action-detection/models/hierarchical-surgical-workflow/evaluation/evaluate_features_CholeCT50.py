import os
import sys
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

class InstrumentFeatureExtractor(nn.Module):
    def __init__(self, yolo_path):
        super().__init__()
        # YOLO Model initialisieren
        self.yolo = YOLO(yolo_path)
        self.yolo_model = self.yolo.model.model
        self.feature_layer = 8  
        
        # YOLO Training deaktivieren
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.yolo_model.eval()
        
    def forward(self, x):
        features = None
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
                    break
        return features
    
def visualize_feature_space(features_array, metadata, save_dir):
    """
    Visualisiert den Feature Space mittels t-SNE
    """
    print("\nBeginne Feature Space Visualisierung...")
    
    # Features vorbereiten
    # Reshape features zu 2D Array (n_samples, n_features)
    n_samples = features_array.shape[0]
    flattened_features = features_array.reshape(n_samples, -1)
    
    # t-SNE Dimensionsreduktion
    print("Führe t-SNE Dimensionsreduktion durch...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(flattened_features)
    
    # Plot erstellen
    plt.figure(figsize=(12, 8))
    
    # Verschiedene Videos mit unterschiedlichen Farben
    unique_videos = list(set(metadata['video_names']))
    colors = sns.color_palette('husl', n_colors=len(unique_videos))
    
    # Plot für jedes Video
    for idx, video in enumerate(unique_videos):
        mask = [v == video for v in metadata['video_names']]
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=video,
            alpha=0.6,
            s=50
        )
    
    plt.title('Feature Space Visualization (pre-Domain Adaptation)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Plot speichern
    plt.savefig(save_dir / 'feature_space_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisierung gespeichert in: {save_dir}/feature_space_visualization.png")
    
    # Zusätzliche Feature-Statistiken berechnen
    print("\nFeature-Statistiken:")
    feature_means = np.mean(flattened_features, axis=0)
    feature_stds = np.std(flattened_features, axis=0)
    print(f"Durchschnittliche Feature-Aktivierung: {np.mean(feature_means):.4f}")
    print(f"Durchschnittliche Feature-Standardabweichung: {np.mean(feature_stds):.4f}")
    
    # Cluster-Analyse
    from sklearn.metrics import silhouette_score
    try:
        silhouette_avg = silhouette_score(features_2d, metadata['video_names'])
        print(f"Silhouette Score: {silhouette_avg:.4f}")
    except:
        print("Silhouette Score konnte nicht berechnet werden")

def extract_cholect50_features():
    # Pfade setzen
    current_dir = Path(__file__).resolve().parent
    hierarchical_dir = current_dir.parent
    yolo_path = hierarchical_dir / "Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
    dataset_path = Path("/data/Bartscht/CholecT50")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Feature Extractor initialisieren
    feature_extractor = InstrumentFeatureExtractor(yolo_path).to(device)
    
    # Bildtransformationen
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Listen für Features und Metadata initialisieren
    all_features = []
    all_video_names = []
    all_frame_numbers = []
    
    print("\nExtrahiere Features aus CholecT50...")
    
    # Verarbeite die spezifizierten Videos
    for video in ["VID92", "VID96", "VID103", "VID110", "VID111"]:
        print(f"\nVerarbeite Video {video}...")
        video_folder = dataset_path / "videos" / video
        
        if not video_folder.exists():
            print(f"Warnung: Video-Ordner {video} nicht gefunden")
            continue
            
        frame_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
        
        for frame_file in tqdm(frame_files, desc=f"Extrahiere Features aus {video}"):
            # Bild laden und vorbereiten
            img_path = video_folder / frame_file
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Features extrahieren
            features = feature_extractor(img_tensor)
            
            # Features für spätere Verarbeitung speichern
            all_features.append(features.cpu().numpy())
            all_video_names.append(video)
            all_frame_numbers.append(int(frame_file.split('.')[0]))
    
    # Features zu einem Array zusammenfügen
    features_array = np.concatenate(all_features, axis=0)
    
    # Metadata erstellen
    metadata = {
        'video_names': all_video_names,
        'frame_numbers': all_frame_numbers
    }
    
    # Features und Metadata speichern
    save_dir = Path('feature_spaces')
    save_dir.mkdir(exist_ok=True)
    
    np.savez(
        save_dir / 'cholect50_instrument_features.npz',
        features=features_array,
        video_names=all_video_names,
        frame_numbers=all_frame_numbers
    )
    
    print(f"\nFeatures wurden gespeichert in: {save_dir}/cholect50_instrument_features.npz")
    print(f"Feature Shape: {features_array.shape}")
    
    # Feature Space visualisieren
    visualize_feature_space(features_array, metadata, save_dir)
    
    return features_array, metadata

if __name__ == "__main__":
    features, metadata = extract_cholect50_features()