import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from ultralytics import YOLO
from copy import deepcopy

class DomainAdapter(nn.Module):
    def __init__(self, yolo_path="/data/Bartscht/YOLO/best_v35.pt", yolo_classes=6, target_classes=5):
        super().__init__()
        # YOLO Initialisierung für Inferenz
        yolo = YOLO(yolo_path)
        self.yolo_model = deepcopy(yolo.model.model)
        del yolo  # Original YOLO Instanz löschen
        
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        self.yolo_model.eval()
        self.feature_layer = 10
        
        # Mapping Matrix
        self.register_buffer('mapping_matrix', torch.zeros(yolo_classes, target_classes))
        self.mapping_matrix[0, 0] = 1  # grasper -> grasper
        self.mapping_matrix[1, 2] = 1  # bipolar -> coagulation
        self.mapping_matrix[2, 2] = 1  # hook -> coagulation
        self.mapping_matrix[3, 3] = 1  # scissors -> scissors
        self.mapping_matrix[4, 1] = 1  # clipper -> clipper
        self.mapping_matrix[5, 4] = 1  # irrigator -> suction_irrigation
        
        # Feature Reducer (gleich wie im Training)
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(512, 384, 1),
            nn.BatchNorm2d(384),
            nn.SiLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.SiLU(),
            nn.Conv2d(384, 384, 3, padding=1, groups=4),
            nn.BatchNorm2d(384),
            nn.SiLU(),
            nn.Conv2d(384, 256, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        
        # Domain Classifier (gleich wie im Training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Sequential(
                nn.Linear(384, 384),
                nn.LayerNorm(384),
                nn.ReLU(),
                nn.Linear(384, 384),
                nn.LayerNorm(384)
            ),
            nn.Sequential(
                nn.Linear(384, 96),
                nn.ReLU(),
                nn.Linear(96, 384),
                nn.Sigmoid()
            ),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Instrument Classifier (exakt wie im Training)
        self.instrument_classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.Sigmoid(),
            ),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Sequential(
                nn.Linear(384, 384),
                nn.LayerNorm(384),
                nn.ReLU(),
                nn.Linear(384, 384),
                nn.LayerNorm(384),
            ),
            nn.Linear(384, yolo_classes),
            nn.Sigmoid()
        )

    def extract_features(self, x):
        features = None
        with torch.no_grad():
            for i, layer in enumerate(self.yolo_model):
                x = layer(x)
                if i == self.feature_layer:
                    features = x.clone()
                    break
        return self.feature_reducer(features)

    def forward(self, x, alpha=1.0):
        features = self.extract_features(x)
        domain_pred = self.domain_classifier(features)
        yolo_pred = self.instrument_classifier(features)
        heichole_pred = torch.matmul(yolo_pred, self.mapping_matrix)
        return domain_pred, yolo_pred, heichole_pred

class AdapterEvaluator:
    def __init__(self, adapter_weights_path, yolo_path, test_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_dir = Path(test_dir)
        
        # Klassen-Mapping für HeiChole
        self.heichole_classes = [
            'grasper',
            'clipper', 
            'coagulation',
            'scissors',
            'suction_irrigation'
        ]
        
        print(f"Loading adapter weights from: {adapter_weights_path}")
        print(f"Loading YOLO from: {yolo_path}")
        print(f"Test directory: {test_dir}")
        
        # Model laden
        self.model = self.load_model(adapter_weights_path, yolo_path)
        
        # Transforms wie im Training
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),  # YOLO Eingabegröße
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, adapter_weights_path, yolo_path):
        # Initialisiere Domain Adapter
        model = DomainAdapter(
            yolo_path=yolo_path,
            yolo_classes=6,
            target_classes=5
        ).to(self.device)
        
        # Lade trainierte Adapter Gewichte
        adapter_weights = torch.load(adapter_weights_path)
        model.feature_reducer.load_state_dict(adapter_weights['feature_reducer'])
        model.instrument_classifier.load_state_dict(adapter_weights['instrument_classifier'])
        model.domain_classifier.load_state_dict(adapter_weights['domain_classifier'])
        
        model.eval()
        return model

    def load_ground_truth(self, video_id):
        json_path = self.test_dir / "Labels" / f"{video_id}.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frame_annotations = defaultdict(lambda: {
            'instruments': defaultdict(int)
        })
        
        for frame_num, frame_data in data['frames'].items():
            frame_number = int(frame_num)
            instruments = frame_data.get('instruments', {})
            for instr_name, present in instruments.items():
                frame_annotations[frame_number]['instruments'][instr_name] = 1 if present > 0 else 0
                
        return frame_annotations

    def evaluate_frame(self, img_path):
        img = Image.open(img_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass durch den Adapter
            domain_pred, yolo_pred, heichole_pred = self.model(img_tensor)
            
            # Konvertiere zu Wahrscheinlichkeiten
            probs = heichole_pred.squeeze().cpu().numpy()
            
            return {
                'probabilities': probs,
                'domain_score': domain_pred.item()
            }

    def evaluate_dataset(self, confidence_threshold=0.5):
        # Get all test videos
        test_videos = [f.stem for f in (self.test_dir / "Labels").glob("*.json")]
        print(f"\nEvaluating videos: {test_videos}")
        
        all_predictions = defaultdict(list)
        all_ground_truth = defaultdict(list)
        domain_scores = []
        
        for video_id in tqdm(test_videos, desc="Evaluating videos"):
            # Lade Ground Truth
            gt = self.load_ground_truth(video_id)
            
            # Evaluiere jedes Frame
            video_dir = self.test_dir / "Videos" / video_id
            for img_path in sorted(video_dir.glob("*.png")):
                frame_num = int(img_path.stem)
                
                # Model Predictions
                results = self.evaluate_frame(img_path)
                domain_scores.append(results['domain_score'])
                
                # Speichere Predictions und Ground Truth
                for idx, prob in enumerate(results['probabilities']):
                    class_name = self.heichole_classes[idx]
                    pred = 1 if prob >= confidence_threshold else 0
                    gt_value = gt[frame_num]['instruments'].get(class_name, 0)
                    
                    all_predictions[class_name].append(pred)
                    all_ground_truth[class_name].append(gt_value)
        
        # Berechne Metriken
        metrics = {}
        for class_name in self.heichole_classes:
            y_true = all_ground_truth[class_name]
            y_pred = all_predictions[class_name]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            accuracy = accuracy_score(y_true, y_pred)
            
            # Zähle positive Vorhersagen und Ground Truth
            total_predictions = sum(y_pred)
            total_ground_truth = sum(y_true)
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'total_ground_truth': total_ground_truth
            }
            
        # Füge durchschnittlichen Domain Score hinzu
        metrics['domain_classification'] = {
            'mean_score': sum(domain_scores) / len(domain_scores),
            'total_frames': len(domain_scores)
        }
        
        return metrics

def main():
    # Konfiguration
    adapter_weights_path = "/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/adapter_weights/adapter_weights.pt"
    yolo_path = "/data/Bartscht/YOLO/best_v35.pt"
    test_dir = "/data/Bartscht/HeiChole/domain_adaptation/test"
    
    # Initialisiere Evaluator
    evaluator = AdapterEvaluator(adapter_weights_path, yolo_path, test_dir)
    
    # Evaluiere
    print("\nStarting evaluation...")
    metrics = evaluator.evaluate_dataset()
    
    # Ausgabe der Ergebnisse
    print("\nEvaluation Results:")
    print("=" * 80)
    print(f"{'Class':20s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'Acc':>8s} {'Pred':>8s} {'GT':>8s}")
    print("-" * 80)
    
    mean_metrics = defaultdict(float)
    for class_name in evaluator.heichole_classes:
        class_metrics = metrics[class_name]
        print(f"{class_name:20s} "
              f"{class_metrics['f1']:8.4f} "
              f"{class_metrics['precision']:8.4f} "
              f"{class_metrics['recall']:8.4f} "
              f"{class_metrics['accuracy']:8.4f} "
              f"{class_metrics['total_predictions']:8d} "
              f"{class_metrics['total_ground_truth']:8d}")
        
        # Berechne Durchschnitt (ohne total_predictions und total_ground_truth)
        for metric in ['precision', 'recall', 'f1', 'accuracy']:
            mean_metrics[metric] += class_metrics[metric]
    
    # Mittlere Metriken
    print("\nMean Metrics:")
    print("-" * 80)
    n_classes = len(evaluator.heichole_classes)
    for metric in ['precision', 'recall', 'f1', 'accuracy']:
        print(f"Mean {metric:8s}: {mean_metrics[metric]/n_classes:.4f}")
        
    # Domain Classification Results
    domain_metrics = metrics['domain_classification']
    print(f"\nDomain Classification:")
    print("-" * 80)
    print(f"Mean Domain Score: {domain_metrics['mean_score']:.4f}")
    print(f"Total Frames Evaluated: {domain_metrics['total_frames']}")

if __name__ == "__main__":
    main()