Active Learning Pipeline for Surgical Instrument Detection
This module implements an active learning pipeline for improving surgical instrument detection in laparoscopic videos using YOLOv8, specifically designed for the CholecT50 dataset.
Dataset
This implementation uses the CholecT50 dataset:
@article{nwoye2022rendezvous,
  title={Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos},
  author={Nwoye, C.I. and Yu, T. and Gonzalez, C. and Seeliger, B. and Mascagni, P. and Mutter, D. and Marescaux, J. and Padoy, N.},
  journal={Medical Image Analysis, Elsevier},
  year={2022}
}
The initial bounding box annotations for localization training were derived from:
@inproceedings{jin2018tool,
  title={Tool detection and operative skill assessment in surgical videos using region-based convolutional neural networks},
  author={Jin, Amy and Yeung, Serena and Jopling, Jeffrey and Krause, Jonathan and Azagury, Dan and Milstein, Arnold and Fei-Fei, Li},
  booktitle={2018 IEEE winter conference on applications of computer vision (WACV)},
  pages={691--699},
  year={2018},
  organization={IEEE}
}
Prerequisites:
Python 3.8+
Ultralytics YOLO
OpenCV
PyYAML
CVAT.ai account

Configuration
The active learning pipeline is configured through instrument_config.yaml
Active Learning Workflow
1. Generate Predictions
Run the model predictor script:
python src/active_learning/model_predictor.py --config config/active_learning/instrument_config.yaml
2. Annotation Refinement in CVAT

Create new task in CVAT.ai
Upload target video frames (e.g., VID26)
Configure instrument labels
Import predictions via "Upload annotations", (in CVAT select annotation format YOLO 1.1)
Refine predictions
Export corrected annotations

3. Model Retraining
Retrain the model using the refined annotations to improve detection performance.
