import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TOOL_MAPPING = {
    0: 'grasper', 1: 'bipolar', 2: 'hook', 
    3: 'scissors', 4: 'clipper', 5: 'irrigator'
}


IGNORED_INSTRUMENTS = {
    6: 'specimen_bag'  # Index: Name der zu ignorierenden Instrumente
}

class FeatureAlignmentHead(nn.Module):
    """
    Neural network head for feature alignment and binary classification.
    Uses a shared feature space for both domain adaptation and classification tasks.
    """
    def __init__(self, num_classes):
        super().__init__()
        # Shared feature extractor
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 256, 1),  # Reduce channel dimension
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten()
        )
        
        # Domain classifier with gradient reversal for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Binary instrument classifier
        self.instrument_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x, alpha=1.0):
        features = self.feature_reducer(x)
        # For evaluation, we only need the class predictions
        domain_pred = self.domain_classifier(features)
        class_pred = self.instrument_classifier(features)
        return domain_pred, class_pred


class InstrumentDetector:
    def __init__(self, yolo_model, alignment_head):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = yolo_model.to(self.device)
        self.alignment_head = alignment_head.to(self.device)
        self.feature_localization = FeatureLocalizationModule()
        
        # Move YOLO model layers to device once during initialization
        for layer in self.yolo_model.model.model:
            layer.to(self.device)
    
    def extract_features(self, img):
        """Extract features from YOLO's backbone"""
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        x = img.to(self.device)
        features = None
        
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model.model.model):
                x = layer(x)
                if i == 10:  # C2PSA layer
                    features = x
                    break
        return features
    
    def create_results_container(self):
        """Creates a container for our modified results"""
        return type('Results', (), {
            'boxes': [],
            'orig_shape': None,
            'names': None
        })
    
    def __call__(self, img):
        """Prozessiert das Bild parallel durch YOLO und Alignment Head"""
        img = img.to(self.device)
        
        # 1. Feature Extraktion (wird von beiden Heads genutzt)
        features = self.extract_features(img)
        
        # 2. Parallel Processing
        # a) YOLO Detektionen
        yolo_results = self.yolo_model(img)
        
        # b) Alignment Head Vorhersagen
        with torch.no_grad():
            _, refined_preds = self.alignment_head(features)
        
        # Debug Output für Alignment Head
        predicted_classes = (refined_preds > 0.5).squeeze()
        for idx, is_present in enumerate(predicted_classes):
            if is_present:
                print(f"Alignment Head detects {TOOL_MAPPING[idx]}")
        
        # 3. Kombiniere Ergebnisse
        modified_results = []
        result_container = self.create_results_container()
        boxes_list = []
        
        # Verarbeite YOLO Detektionen
        for result in yolo_results:
            for box in result.boxes:
                original_cls = int(box.cls)
                if original_cls in IGNORED_INSTRUMENTS:
                    continue
                    
                yolo_conf = float(box.conf)
                align_conf = float(refined_preds[0][original_cls])
                
                # Kombiniere Konfidenzen (z.B. Durchschnitt oder Maximum)
                combined_conf = max(yolo_conf, align_conf)
                
                box_dict = {
                    'xyxy': box.xyxy[0],
                    'cls': torch.tensor([original_cls], device=self.device),
                    'conf': torch.tensor([combined_conf], device=self.device),
                    'yolo_conf': torch.tensor([yolo_conf], device=self.device),
                    'align_conf': torch.tensor([align_conf], device=self.device)
                }
                boxes_list.append(box_dict)
        
        # Suche nach zusätzlichen Detektionen vom Alignment Head
        new_detections = self.feature_localization.detect_missed_instruments(
            features, 
            self.alignment_head,
            refined_preds,
            img.shape[2:]
        )
        
        # Füge neue Detektionen hinzu wenn sie nicht überlappen
        for new_det in new_detections:
            is_unique = True
            for existing_box in boxes_list:
                if self.calculate_iou(new_det['xyxy'][0], existing_box['xyxy']) > 0.3:  # Reduzierter IOU-Schwellwert
                    is_unique = False
                    break
            
            if is_unique:
                boxes_list.append(new_det)
                print(f"Added new detection from Alignment Head: {TOOL_MAPPING[int(new_det['cls'])]}")
        
        # Update Result Container
        result_container.boxes = boxes_list
        if yolo_results:  # Übernehme Metadaten vom ersten YOLO-Ergebnis
            result_container.orig_shape = yolo_results[0].orig_shape
            result_container.names = yolo_results[0].names
        
        modified_results.append(result_container)
        
        return modified_results
    
class FeatureLocalizationModule:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute_class_activation_map(self, features, alignment_head, class_idx):
        """
        Berechnet die Class Activation Map (CAM) für eine bestimmte Klasse
        
        Args:
            features (torch.Tensor): Feature map aus der C2PSA layer [1, C, H, W]
            alignment_head: Der trainierte Alignment Head
            class_idx (int): Index der Instrumentenklasse
            
        Returns:
            torch.Tensor: Activation map für die spezifische Klasse [H, W]
        """
        # Get weights from the last layer of instrument classifier
        weights = alignment_head.instrument_classifier[-2].weight[class_idx]
        
        # Get feature maps before global average pooling
        feature_maps = features[0]  # Remove batch dimension
        
        # Compute weighted sum of feature maps
        cam = torch.zeros(feature_maps.shape[1:], device=self.device)
        for w, feat_map in zip(weights, feature_maps):
            cam += w * feat_map
            
        # Apply ReLU to focus on positive activations
        cam = F.relu(cam)
        
        # Normalize the CAM
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam
    
    def find_activation_regions(self, cam, original_size):
        """
        Findet Regionen mit hoher Aktivierung in der CAM
        
        Args:
            cam (torch.Tensor): Class Activation Map [H, W]
            original_size (tuple): Original image size (H, W)
            
        Returns:
            list: Liste von Bounding Boxes im Format [x1, y1, x2, y2]
        """
        # Konvertiere zu numpy für Verarbeitung
        cam_detached = cam.detach()
        cam_cpu = cam_detached.cpu()
        cam_np = cam_cpu.numpy()
        
        # Binärisiere die CAM mit Threshold
        binary_map = (cam_np > self.threshold).astype(np.uint8)
        
        # Finde zusammenhängende Regionen
        from scipy import ndimage
        labeled_map, num_features = ndimage.label(binary_map)
        
        boxes = []
        for region_id in range(1, num_features + 1):
            # Get region coordinates
            region_coords = np.where(labeled_map == region_id)
            if len(region_coords[0]) < 10:  # Filter kleine Regionen
                continue
                
            # Compute bounding box
            y1, x1 = np.min(region_coords[0]), np.min(region_coords[1])
            y2, x2 = np.max(region_coords[0]), np.max(region_coords[1])
            
            # Scale coordinates to original image size
            h_scale = original_size[0] / cam.shape[0]
            w_scale = original_size[1] / cam.shape[1]
            
            box = [
                int(x1 * w_scale),
                int(y1 * h_scale),
                int(x2 * w_scale),
                int(y2 * h_scale)
            ]
            boxes.append(box)
            
        return boxes
    
    def detect_missed_instruments(self, features, alignment_head, refined_preds, original_size):
        """
        Detektiert Instrumente die von YOLO möglicherweise übersehen wurden
        
        Args:
            features (torch.Tensor): Feature map aus der C2PSA layer
            alignment_head: Der trainierte Alignment Head
            refined_preds (torch.Tensor): Vorhersagen des Alignment Heads
            original_size (tuple): Original image size (H, W)
            
        Returns:
            list: Liste von Detektionen im Format {'box': [x1,y1,x2,y2], 'cls': class_idx, 'conf': confidence}
        """
        new_detections = []
        
        # Für jede Klasse mit hoher Konfidenz
        for class_idx, confidence in enumerate(refined_preds[0]):
            if float(confidence) > 0.9:  # Nur bei hoher Konfidenz
                # Berechne CAM für diese Klasse
                cam = self.compute_class_activation_map(features, alignment_head, class_idx)
                
                # Finde Regionen mit hoher Aktivierung
                boxes = self.find_activation_regions(cam, original_size)
                
                # Erstelle Detektionen für gefundene Regionen
                for box in boxes:
                    detection = {
                        'xyxy': torch.tensor([box], device=self.device),
                        'cls': torch.tensor([class_idx], device=self.device),
                        'conf': torch.tensor([float(confidence)], device=self.device)
                    }
                    new_detections.append(detection)
        
        return new_detections