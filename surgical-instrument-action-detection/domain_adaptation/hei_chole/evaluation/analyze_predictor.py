from ultralytics import YOLO
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import traceback
from pathlib import Path
from ultralytics.engine.results import Results
from ultralytics.engine.predictor import BasePredictor

class YOLOInternalDebugger:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        
        self.model = YOLO(model_path)
        self.model.model.eval()
        self.model.model = self.model.model.to(self.device)
        
        # Analyze model components
        print("\n=== Model Components ===")
        print(f"Model type: {type(self.model)}")
        print(f"Model.model type: {type(self.model.model)}")
        
        # Get Detection Head
        self.detector = self.model.model.model[-1]
        print(f"\nDetector type: {type(self.detector)}")
        print(f"Detector attributes: {dir(self.detector)}")
        
        # Create predictor if needed
        if self.model.predictor is None:
            self.predictor = BasePredictor()
            print("\nCreated new BasePredictor")
        else:
            self.predictor = self.model.predictor
            print("\nUsing existing predictor")
            
    def analyze_prediction_process(self, img_path):
        """Analyze the complete prediction process"""
        print("\n=== Analyzing Prediction Process ===")
        
        try:
            # 1. Standard prediction
            img = Image.open(img_path)
            results1 = self.model(img)
            print(f"\n1. Standard prediction results type: {type(results1)}")
            print(f"First result attributes: {results1[0].__dict__.keys()}")
            
            if hasattr(results1[0], 'boxes') and len(results1[0].boxes) > 0:
                print("\nBoxes from standard prediction:")
                print(f"Box coordinates: {results1[0].boxes.xyxy[0]}")
                print(f"Confidence: {results1[0].boxes.conf[0]}")
                print(f"Class: {results1[0].boxes.cls[0]}")
            
            # 2. Manual prediction process
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Print detector info
                print("\nDetector configuration:")
                print(f"Detector strides: {self.detector.stride}")
                print(f"Detector anchors: {self.detector.anchors if hasattr(self.detector, 'anchors') else 'No anchors'}")
                
                # Get model output
                output = self.model.model(img_tensor)
                print(f"\n2. Raw model output shape: {[out.shape for out in output if isinstance(out, torch.Tensor)]}")
                
                # Analyze detector output format
                if isinstance(output, tuple) and len(output) > 0:
                    det_output = output[0]
                    print(f"\nDetector output shape: {det_output.shape}")
                    print(f"Output min: {det_output.min().item():.3f}")
                    print(f"Output max: {det_output.max().item():.3f}")
                    print(f"Output mean: {det_output.mean().item():.3f}")
                    
                    # Show how outputs are structured
                    if len(det_output.shape) == 3:  # [batch, anchors, data]
                        print(f"\nOutput structure:")
                        print(f"Batch size: {det_output.shape[0]}")
                        print(f"Number of detections: {det_output.shape[1]}")
                        print(f"Features per detection: {det_output.shape[2]}")

        except Exception as e:
            print(f"Error in prediction analysis: {str(e)}")
            traceback.print_exc()

def main():
    model_path = str(Path("/home/Bartscht/YOLO/surgical-instrument-action-detection/models/hierarchical-surgical-workflow/Instrument-classification-detection/weights/instrument_detector/best_v35.pt"))
    test_image = str(Path("/data/Bartscht/HeiChole/domain_adaptation/test/Videos/VID08/030300.png"))
    
    debugger = YOLOInternalDebugger(model_path)
    debugger.analyze_prediction_process(test_image)

if __name__ == "__main__":
    main()