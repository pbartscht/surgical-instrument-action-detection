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
    5: {'name': 'irrigator', 'weight': 1.0, 'base_threshold': 0.15}
}

class YOLOLossAdapter:
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.current_loss = None
        
        # Hook für das Haupt-Modell
        self.model.model.register_forward_hook(self._capture_loss)
        
        # Zusätzliche Hooks für Submodule falls nötig
        for module in self.model.model.modules():
            if hasattr(module, 'loss') or 'Loss' in module.__class__.__name__:
                module.register_forward_hook(self._capture_loss)

    def _capture_loss(self, module, input, output):
        """Erweiterte Loss-Erfassung"""
        # Fall 1: Loss im Output als direkter Wert
        if isinstance(output, dict) and 'loss' in output:
            self.current_loss = output['loss']
            return
        
        # Fall 2: Loss als Attribut
        if hasattr(output, 'loss'):
            loss_attr = getattr(output, 'loss')
            if not callable(loss_attr):  # Wenn es ein Wert ist
                self.current_loss = loss_attr
                return
        
        # Fall 3: Loss im Modul
        if hasattr(module, 'loss'):
            loss_attr = getattr(module, 'loss')
            if not callable(loss_attr):  # Wenn es ein Wert ist
                self.current_loss = loss_attr
                return
        
        # Wenn kein direkter Loss-Wert gefunden wurde
        self.current_loss = torch.tensor(0.0)
                
        # Konvertierung zu Tensor
        if self.current_loss is not None:
            if isinstance(self.current_loss, torch.Tensor):
                self.current_loss = self.current_loss.detach()
            elif isinstance(self.current_loss, (int, float)):
                self.current_loss = torch.tensor(self.current_loss)

    def train_step(self, batch, custom_loss_fn=None):
        """Sicherer Trainingsschritt"""
        try:
            # YOLO's forward pass
            results = self.model.train(**batch)  # Nutze train statt __call__
            
            # Stelle sicher, dass wir einen Loss haben
            if self.current_loss is None:
                print("Warning: No YOLO loss captured, using only custom loss")
                yolo_loss = torch.tensor(0.0).to(self.model.device)
            else:
                yolo_loss = self.current_loss
                
            if custom_loss_fn:
                try:
                    custom_loss = custom_loss_fn(results, batch)
                    total_loss = yolo_loss + custom_loss
                    
                    return {
                        'yolo_loss': yolo_loss.item(),
                        'custom_loss': custom_loss.item(),
                        'total_loss': total_loss.item()
                    }
                except Exception as e:
                    print(f"Error in custom loss calculation: {str(e)}")
                    return {'yolo_loss': yolo_loss.item()}
            
            return {'yolo_loss': yolo_loss.item()}
            
        except Exception as e:
            print(f"Error in YOLO training step: {str(e)}")
            # Fallback: Minimaler Loss um Training fortzusetzen
            return {'yolo_loss': 0.0}

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