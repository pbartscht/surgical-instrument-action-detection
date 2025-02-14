import torch
from ultralytics import YOLO
from pathlib import Path
import copy
from PIL import Image
import numpy as np
from torch.distributions import Normal

# Constants
CONFIDENCE_THRESHOLD = 0.1
IMAGE_SIZE = (640, 640)  # Standardgröße für alle Bilder

# Tool Mapping mit Performance-Gewichtung
TOOL_MAPPING = {
    0: {'name': 'grasper', 'weight': 0.3, 'base_threshold': 0.25},
    1: {'name': 'bipolar', 'weight': 1.0, 'base_threshold': 0.15},
    2: {'name': 'hook', 'weight': 0.3, 'base_threshold': 0.25},
    3: {'name': 'scissors', 'weight': 1.0, 'base_threshold': 0.15},
    4: {'name': 'clipper', 'weight': 1.0, 'base_threshold': 0.15},
    5: {'name': 'irrigator', 'weight': 1.0, 'base_threshold': 0.15}
}

class DualModelManager:
    def __init__(self, inference_weights, train_weights=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inference_model = YOLO(inference_weights)
        
        if train_weights is None:
            self.train_model = YOLO(inference_weights)
        else:
            self.train_model = YOLO(train_weights)
            
        self.update_counter = 0
        self.update_frequency = 100

    def predict_with_gaussian_uncertainty(self, image, box):
        """Implementiert Gaussian-basierte Box Uncertainty"""
        # Extrahiere Boxparameter
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Berechne Gaussian Parameter
        mu = torch.tensor([x1, y1, width, height])
        sigma = torch.tensor([
            width * 0.1,  # 10% der Breite als Standardabweichung
            height * 0.1, # 10% der Höhe als Standardabweichung
            width * 0.2,  # 20% für Breite
            height * 0.2  # 20% für Höhe
        ])
        
        # Erstelle Normalverteilung
        distribution = Normal(mu, sigma)
        
        # Berechne Log-Likelihood der Box
        log_prob = distribution.log_prob(mu).mean()
        
        # Normalisiere zu Confidence-Score
        uncertainty = 1 - torch.exp(log_prob).item()
        return uncertainty

    def train_step(self, batch):
        """Erweitertes Training mit Consistency Loss"""
        # Standard Detection Training
        det_loss = self.train_model.train(**batch['detection'])
        
        # Consistency Loss
        if 'consistency' in batch:
            cons_loss = self.calculate_consistency_loss(
                batch['consistency']['mixed_pred'],
                batch['consistency']['source_pred'],
                batch['consistency']['weights']
            )
            total_loss = det_loss + 0.5 * cons_loss
        else:
            total_loss = det_loss
            
        self.update_counter += 1
        return total_loss

    def calculate_consistency_loss(self, mixed_pred, source_pred, weights):
        """Berechnet gewichteten Consistency Loss"""
        loss = 0
        for mp, sp, w in zip(mixed_pred, source_pred, weights):
            # Berechne L2 Loss zwischen Predictions
            box_loss = torch.nn.functional.mse_loss(
                mp['boxes'], sp['boxes']
            )
            # Gewichte mit Klassengewicht
            loss += w * box_loss
        return loss

    def _update_inference_model(self):
        """Aktualisiert Inference-Modell"""
        state_dict = copy.deepcopy(self.train_model.model.state_dict())
        self.inference_model.model.load_state_dict(state_dict)
        self.update_counter = 0
        print("\nInference model updated with new weights")

    def save_models(self, save_dir):
        """Speichert beide Modelle"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        self.inference_model.save(save_dir / 'inference_model.pt')
        self.train_model.save(save_dir / 'train_model.pt')

class YOLOUtils:
    @staticmethod
    def convert_to_yolo_format(box, image_width, image_height):
        """Konvertiert Box-Koordinaten ins YOLO-Format"""
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Normalisierung
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        
        return x_center, y_center, width, height

    @staticmethod
    def get_class_specific_threshold(class_id, progress_ratio):
        """Berechnet klassenspezifische Confidence Schwellen"""
        tool_info = TOOL_MAPPING[class_id]
        base_threshold = tool_info['base_threshold']
        weight = tool_info['weight']
        
        # Progressiver Threshold
        threshold = base_threshold * (1 + progress_ratio * weight)
        return min(threshold, 0.9)