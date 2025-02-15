import torch
from ultralytics import YOLO
from pathlib import Path
import copy
import numpy as np

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
    5: {'name': 'irrigator', 'weight': 1.0, 'base_threshold': 0.15},
    6: {'name': 'specimen_bag', 'weight': 0.1, 'base_threshold': 0.3}
}

class YOLOLossAdapter:
    def __init__(self, yolo_model):
        self.model = yolo_model
        # Überschreibe die Standard-Trainingsmethode
        self.model.train = self.custom_train
        
        # Basis-Setup
        self.device = next(self.model.parameters()).device
        self.detect = self.model.model.model[-1]  # Detection Layer
        self.criterion = self.model.model.loss    # Loss Function
        
        # Custom Training Parameter
        self.num_epochs = 50  # Unsere definierte Epochenzahl
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Optimizer Setup mit verschiedenen Parametergruppen
        pg0, pg1, pg2 = [], [], []  # Parameter Gruppen
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
                pg2.append(v.bias)    # Biases
            if isinstance(v, torch.nn.BatchNorm2d):
                pg0.append(v.weight)  # BatchNorm Weights
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
                pg1.append(v.weight)  # Rest der Weights

        self.optimizer = torch.optim.Adam(pg0, lr=0.01)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0005})
        self.optimizer.add_param_group({'params': pg2})
        
        # Loss tracking
        self.current_loss = None
        self.loss_history = []
        
    def custom_train(self, **kwargs):
        """Überschreibt die Standard YOLO Trainingsmethode"""
        # Verhindere Standard YOLO Training
        self.model.model.train()
        return self.model

    def train_step(self, batch, custom_loss_fn=None):
        """Erweiterter Trainingsschritt mit Custom Loss Support"""
        try:
            self.model.model.train()  # Explizit Trainingsmodus aktivieren
            self.optimizer.zero_grad()
            
            # Batch zu Device verschieben
            images = batch['images'].to(self.device)
            targets = self._convert_targets(batch['labels'])
            
            # Forward pass
            pred = self.model(images)
            loss = self.criterion(pred, targets)
            
            # Custom loss wenn verfügbar
            if custom_loss_fn:
                try:
                    custom_loss = custom_loss_fn(pred, batch)
                    loss = loss + custom_loss
                except Exception as e:
                    print(f"Warning: Custom loss calculation failed: {str(e)}")
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Loss tracking
            self.current_loss = loss.detach()
            self.loss_history.append(float(self.current_loss))
            
            return {
                'loss': float(self.current_loss),
                'pred': pred
            }
            
        except Exception as e:
            print(f"Error in YOLO training step: {str(e)}")
            return {'loss': 0.0, 'pred': None}

    def _convert_targets(self, targets):
        """Konvertiert Targets in YOLO-Format"""
        converted = []
        for target in targets:
            if isinstance(target, (list, tuple)):
                boxes = []
                for t in target:
                    box = {
                        'cls': torch.tensor([t['class']]).to(self.device),
                        'boxes': torch.tensor(t['box']).to(self.device),
                        'conf': torch.tensor(1.0).to(self.device)
                    }
                    boxes.append(box)
                converted.append(boxes)
            else:
                box = {
                    'cls': torch.tensor([target['class']]).to(self.device),
                    'boxes': torch.tensor(target['box']).to(self.device),
                    'conf': torch.tensor(1.0).to(self.device)
                }
                converted.append([box])
        
        return converted

    def save_state(self, path):
        """Speichert Model und Optimizer State"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'loss_history': self.loss_history,
            'best_loss': self.best_loss
        }, path)

    def load_state(self, path):
        """Lädt gespeicherten State"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.loss_history = checkpoint['loss_history']
        self.best_loss = checkpoint['best_loss']

class DualModelManager:
    def __init__(self, inference_weights, train_weights=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize YOLO models
        self.inference_model = YOLO(inference_weights)
        if train_weights is None:
            self.train_model = YOLO(inference_weights)
        else:
            self.train_model = YOLO(train_weights)
            
        # Wrap models mit Loss Adapter
        self.inference_adapter = YOLOLossAdapter(self.inference_model)
        self.train_adapter = YOLOLossAdapter(self.train_model)
            
        self.update_counter = 0
        self.update_frequency = 100

    def train_step(self, batch, custom_loss_fn=None):
        """Training step mit Loss Adapter"""
        return self.train_adapter.train_step(batch, custom_loss_fn)

    def predict(self, images):
        """Prediction mit Inference Model"""
        with torch.no_grad():
            return self.inference_model(images)

    def _update_inference_model(self):
        """Aktualisiert Inference-Modell"""
        state_dict = copy.deepcopy(self.train_model.model.state_dict())
        self.inference_model.model.load_state_dict(state_dict)
        # Update auch den Loss Adapter
        self.inference_adapter = YOLOLossAdapter(self.inference_model)
        self.update_counter = 0
        print("\nInference model updated with new weights")

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