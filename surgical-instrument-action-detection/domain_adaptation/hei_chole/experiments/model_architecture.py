import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Move YOLO model layers to device once during initialization
        for layer in self.yolo_model.model.model:
            layer.to(self.device)
    
    def extract_features(self, img):
        """Extract features from YOLO's backbone"""
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        # Ensure input is on correct device
        x = img.to(self.device)
        features = None
        
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model.model.model):
                x = layer(x)  # Layer is already on correct device from __init__
                if i == 10:  # C2PSA layer
                    features = x
                    break
                    
        return features
    
    def __call__(self, img):
        """
        Process image through both YOLO and alignment head
        Returns format compatible with original YOLO output
        """
        # Ensure input is on correct device
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        img = img.to(self.device)
        
        # Get base YOLO predictions
        yolo_results = self.yolo_model(img)
        
        # Extract features for alignment head
        features = self.extract_features(img)
        
        # Get refined predictions from alignment head
        with torch.no_grad():
            _, refined_preds = self.alignment_head(features)
        
        # Modify YOLO results with refined predictions
        for i, box in enumerate(yolo_results[0].boxes):
            original_cls = int(box.cls)
            refined_conf = float(refined_preds[0][original_cls])
            box.conf = box.conf * refined_conf  # Combine confidences
        
        return yolo_results