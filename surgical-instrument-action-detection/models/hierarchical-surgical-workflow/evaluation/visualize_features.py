import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
try:
    # Try GPU version first
    from cuml.manifold import TSNE
    import cupy as cp
    USE_GPU = True
except ImportError:
    # Fallback to CPU version
    from sklearn.manifold import TSNE
    USE_GPU = False
from sklearn.metrics import silhouette_score

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
    try:
        silhouette_avg = silhouette_score(features_2d, metadata['video_names'])
        print(f"Silhouette Score: {silhouette_avg:.4f}")
    except:
        print("Silhouette Score konnte nicht berechnet werden")

def visualize_shuffled_features(max_samples=2000):
    """
    Lädt die gespeicherten Features, shuffled sie und erstellt eine neue Visualisierung
    mit GPU-Beschleunigung falls verfügbar, sonst CPU
    """
    # Pfade setzen
    save_dir = Path('/home/Bartscht/YOLO/surgical-instrument-action-detection/models/hierarchical-surgical-workflow/evaluation')
    npz_path = save_dir / 'feature_spaces/cholect50_instrument_features.npz'
    
    # NPZ Datei laden
    print("Lade NPZ-Datei...")
    data = np.load(npz_path)
    
    features_array = data['features']
    metadata = {
        'video_names': data['video_names'],
        'frame_numbers': data['frame_numbers']
    }
    
    print("Geladene Feature Shape:", features_array.shape)
    
    # Features shufflen
    print("\nShuffling Features...")
    total_samples = len(features_array)
    if total_samples > max_samples:
        print(f"Reduziere Samples von {total_samples} auf {max_samples} für Verarbeitung...")
        indices = np.random.choice(total_samples, max_samples, replace=False)
        features_array = features_array[indices]
        metadata = {
            'video_names': [metadata['video_names'][i] for i in indices],
            'frame_numbers': [metadata['frame_numbers'][i] for i in indices]
        }
    
    random_indices = np.random.permutation(len(features_array))
    shuffled_features = features_array[random_indices]
    shuffled_metadata = {
        'video_names': [metadata['video_names'][i] for i in random_indices],
        'frame_numbers': [metadata['frame_numbers'][i] for i in random_indices]
    }
    
    # Features vorbereiten
    print("\nFeatures vorbereiten...")
    n_samples = shuffled_features.shape[0]
    flattened_features = shuffled_features.reshape(n_samples, -1)
    
    # t-SNE durchführen (GPU oder CPU)
    if USE_GPU:
        print("\nVerwende GPU für t-SNE...")
        gpu_features = cp.asarray(flattened_features)
        tsne = TSNE(n_components=2, 
                    random_state=42, 
                    perplexity=min(30, len(gpu_features) - 1))
        features_2d = tsne.fit_transform(gpu_features)
        features_2d = cp.asnumpy(features_2d)
        # GPU Speicher freigeben
        del gpu_features
        cp.get_default_memory_pool().free_all_blocks()
    else:
        print("\nVerwende CPU für t-SNE...")
        tsne = TSNE(n_components=2,
                    random_state=42,
                    perplexity=min(30, len(flattened_features) - 1))
        features_2d = tsne.fit_transform(flattened_features)
    
    # Plot erstellen
    plt.figure(figsize=(12, 8))
    
    # Verschiedene Videos mit unterschiedlichen Farben
    unique_videos = list(set(shuffled_metadata['video_names']))
    colors = sns.color_palette('husl', n_colors=len(unique_videos))
    
    # Plot für jedes Video
    for idx, video in enumerate(unique_videos):
        mask = [v == video for v in shuffled_metadata['video_names']]
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=video,
            alpha=0.6,
            s=50
        )
    
    plt.title('Feature Space Visualization (Shuffled)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Plot speichern
    plt.savefig(save_dir / 'feature_space_shuffled.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualisierung gespeichert in: {save_dir}/feature_space_shuffled.png")
    
    # Feature-Statistiken berechnen
    print("\nFeature-Statistiken (nach Shuffling):")
    feature_means = np.mean(flattened_features, axis=0)
    feature_stds = np.std(flattened_features, axis=0)
    print(f"Durchschnittliche Feature-Aktivierung: {np.mean(feature_means):.4f}")
    print(f"Durchschnittliche Feature-Standardabweichung: {np.mean(feature_stds):.4f}")
    
    # Cluster-Analyse
    try:
        silhouette_avg = silhouette_score(features_2d, shuffled_metadata['video_names'])
        print(f"Silhouette Score (nach Shuffling): {silhouette_avg:.4f}")
    except:
        print("Silhouette Score konnte nicht berechnet werden")


def main():
    # Pfad zur npz-Datei
    npz_path = Path('/home/Bartscht/YOLO/surgical-instrument-action-detection/models/hierarchical-surgical-workflow/evaluation/feature_spaces/cholect50_instrument_features.npz')
    save_dir = Path('/home/Bartscht/YOLO/surgical-instrument-action-detection/models/hierarchical-surgical-workflow/evaluation')
    
    # NPZ-Datei laden
    print("Lade NPZ-Datei...")
    data = np.load(npz_path)
    
    # Daten extrahieren
    features_array = data['features']
    metadata = {
        'video_names': data['video_names'],
        'frame_numbers': data['frame_numbers']
    }
    
    print(f"Geladene Feature Shape: {features_array.shape}")
    
    # Feature Space visualisieren
    #visualize_feature_space(features_array, metadata, save_dir)
    print("\nErstelle geshuffelte Visualisierung...")
    visualize_shuffled_features()

if __name__ == "__main__":
    visualize_shuffled_features()