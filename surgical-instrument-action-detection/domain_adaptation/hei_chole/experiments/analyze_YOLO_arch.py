import sys
from pathlib import Path
import torch
from ultralytics import YOLO
from pprint import pprint

# Pfadstruktur aus Ihrem Code
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
hierarchical_dir = project_root / "models" / "hierarchical-surgical-workflow"

class YOLOAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        try:
            self.model = YOLO(str(self.model_path))
            print("✓ YOLO model loaded successfully")
            return self.model
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            raise
            
    def analyze_model_structure(self):
        """Analysiert die Struktur des YOLO Models"""
        print("\n=== YOLO Model Analysis ===")
        
        # Basis Model Info
        print("\n1. Basic Model Info:")
        print(f"Model Type: {type(self.model)}")
        print(f"Model Name: {self.model.name if hasattr(self.model, 'name') else 'Unknown'}")
        print(f"Task: {self.model.task if hasattr(self.model, 'task') else 'Unknown'}")
        
        # Backbone Analysis
        print("\n2. Backbone Analysis:")
        try:
            backbone = self.model.model.model[0] if hasattr(self.model, 'model') else None
            print(f"Backbone structure:")
            print(backbone)
            
            # Versuche Feature Dimensionen zu bekommen
            if backbone is not None:
                # Dummy forward pass für Feature Dimensionen
                dummy_input = torch.randn(1, 3, 512, 512)  # Standard YOLO input size
                with torch.no_grad():
                    try:
                        features = backbone(dummy_input)
                        if isinstance(features, torch.Tensor):
                            print(f"\nFeature dimensions: {features.shape}")
                        elif isinstance(features, (list, tuple)):
                            print("\nFeature dimensions (multiple outputs):")
                            for i, feat in enumerate(features):
                                print(f"Output {i}: {feat.shape}")
                    except Exception as e:
                        print(f"Could not determine feature dimensions: {str(e)}")
            
        except Exception as e:
            print(f"Could not access backbone directly: {str(e)}")
        
        # Layer Analysis
        print("\n3. Layer Analysis:")
        try:
            print("Model layers:")
            model_layers = self.model.model.model if hasattr(self.model.model, 'model') else None
            if model_layers is not None:
                for i, layer in enumerate(model_layers):
                    print(f"\nLayer {i}:")
                    print(f"Type: {type(layer)}")
                    print(f"Structure: {layer}")
        except Exception as e:
            print(f"Could not analyze layers: {str(e)}")

def main():
    # Modelpfad aus ModelLoader
    model_path = "/home/Bartscht/YOLO/surgical-instrument-action-detection/models/hierarchical-surgical-workflow/Instrument-classification-detection/weights/instrument_detector/best_v35.pt"
    
    # Analyzer erstellen und Model laden
    analyzer = YOLOAnalyzer(model_path)
    model = analyzer.load_model()
    
    # Analyse durchführen
    analyzer.analyze_model_structure()

if __name__ == "__main__":
    main()