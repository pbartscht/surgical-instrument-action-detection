import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import math
import logging
import random
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from ultralytics.utils import LOGGER, SETTINGS

# Configuration
CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.25,
    'GAMMA_THRESHOLD': 0.5,
    'BATCH_SIZE': 8,
    'EPOCHS': 50,
    'LEARNING_RATE': 0.001,
    'SEED': 42,
    'IMAGE_SIZE': 512,
    'WORKERS': 4
}

# Paths
PATHS = {
    'YOLO_MODEL': "/data/Bartscht/YOLO/best_v35.pt",
    'SOURCE_DATA': "/data/Bartscht/YOLO",
    'TARGET_DATA': "/data/Bartscht/HeiChole/domain_adaptation/test",
    'OUTPUT_DIR': "/data/Bartscht/confmix_training"
}

def collate_fn(batch):
    """Custom collate function to handle different sized labels"""
    images = []
    labels = []
    paths = []
    
    for item in batch:
        images.append(item['image'])
        if item['labels'] is not None:
            labels.append(item['labels'])
        else:
            labels.append(torch.zeros((0, 5)))  # Empty label tensor
        paths.append(item['path'])
    
    # Stack images
    images = torch.stack(images)
    
    return {
        'image': images,
        'labels': labels,  # Keep as list of tensors
        'path': paths
    }

class DomainDataset(Dataset):
    def __init__(self, data_path, is_source=True, image_size=512):
        self.is_source = is_source
        self.image_paths = []
        self.label_paths = []
        self.image_size = image_size
        
        if is_source:
            img_path = Path(data_path) / "images" / "train"
            label_path = Path(data_path) / "labels" / "train"
            self.image_paths.extend(list(img_path.glob("*.png")))
            self.label_paths = [label_path / f"{img.stem}.txt" for img in self.image_paths]
        else:
            videos_path = Path(data_path) / "Videos"
            for video_folder in videos_path.iterdir():
                if video_folder.is_dir():
                    self.image_paths.extend(list(video_folder.glob("*.png")))

    def __len__(self):
        return len(self.image_paths)

    def preprocess_image(self, image):
        """Preprocess image to match YOLO requirements"""
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image.contiguous()  # Make tensor contiguous in memory

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess_image(image)
        
        if self.is_source:
            label_path = self.label_paths[idx]
            if label_path.exists():
                with open(label_path, 'r') as f:
                    labels = []
                    for line in f:
                        try:
                            values = [float(x) for x in line.strip().split()]
                            labels.append(values)
                        except ValueError:
                            continue
                    labels = torch.tensor(labels) if labels else torch.zeros((0, 5))
            else:
                labels = torch.zeros((0, 5))
        else:
            labels = torch.zeros((0, 5))
            
        return {
            'image': image,
            'labels': labels,
            'path': str(img_path)
        }

class ConfMixTrainer:
    def __init__(self, model_path, source_path, target_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logging()
        
        # Prevent auto-training by modifying YOLO settings
        SETTINGS['resume'] = False
        SETTINGS['autoresume'] = False
        
        # Initialize YOLO model without starting training
        self.logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        
        # Disable default trainer and set custom configuration
        self.model.trainer = None
        
        # Initialize datasets
        self.source_dataset = DomainDataset(source_path, is_source=True, image_size=CONFIG['IMAGE_SIZE'])
        self.target_dataset = DomainDataset(target_path, is_source=False, image_size=CONFIG['IMAGE_SIZE'])
        
        # Initialize dataloaders with custom collate_fn
        self.source_loader = DataLoader(
            self.source_dataset,
            batch_size=CONFIG['BATCH_SIZE'],
            shuffle=True,
            num_workers=CONFIG['WORKERS'],
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        self.target_loader = DataLoader(
            self.target_dataset,
            batch_size=CONFIG['BATCH_SIZE'],
            shuffle=True,
            num_workers=CONFIG['WORKERS'],
            pin_memory=True,
            collate_fn=collate_fn
        )

        # Set model configuration
        self.model.overrides = {
            'epochs': CONFIG['EPOCHS'],
            'batch': CONFIG['BATCH_SIZE'],
            'optimizer': 'Adam',
            'lr0': CONFIG['LEARNING_RATE'],
            'device': self.device,
            'model': model_path,
            'data': None,
            'mode': None,
            'task': 'detect',
            'imgsz': CONFIG['IMAGE_SIZE'],
            'save': True,
            'save_period': 5,
            'cache': False,
            'workers': CONFIG['WORKERS'],
            'project': PATHS['OUTPUT_DIR'],
            'name': 'confmix_training',
            'exist_ok': True,
            'pretrained': True,
            'amp': True,
            'verbose': True
        }
        
        # Set model to train mode and move to device
        self.model.model.train()
        self.model.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.model.parameters(),
            lr=CONFIG['LEARNING_RATE']
        )
        
        # Training state
        self.current_iteration = 0
        self.total_iterations = len(self.source_loader) * CONFIG['EPOCHS']
        self.best_loss = float('inf')
        
        self.logger.info("ConfMix trainer initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(PATHS['OUTPUT_DIR'], exist_ok=True)
        log_file = Path(PATHS['OUTPUT_DIR']) / 'training.log'
        
        logger = logging.getLogger('ConfMixTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger

    def get_pseudo_detections(self, predictions, conf_thres=None):
        """Get pseudo detections with confidence filtering and NMS"""
        if conf_thres is None:
            conf_thres = CONFIG['CONFIDENCE_THRESHOLD']
            
        pseudo_dets = []
        
        # Process each prediction in the batch
        for pred in predictions:
            # Get the boxes from the prediction
            boxes = pred[0]  # First tensor contains box predictions
            
            if len(boxes) > 0:
                # Extract confidence scores (5th column)
                confs = boxes[:, 4]
                # Apply confidence threshold
                conf_mask = confs > conf_thres
                filtered_boxes = boxes[conf_mask]
                
                if len(filtered_boxes) > 0:
                    # Get coordinates and class predictions
                    coords = filtered_boxes[:, :4]  # First 4 columns are coordinates
                    scores = filtered_boxes[:, 4]   # 5th column is objectness score
                    cls_scores = filtered_boxes[:, 5:]  # Remaining columns are class scores
                    cls_ids = cls_scores.argmax(dim=1)
                    
                    # Apply NMS
                    keep = torchvision.ops.nms(coords, scores, iou_threshold=0.5)
                    
                    pseudo_dets.append({
                        'boxes': coords[keep],
                        'confs': scores[keep],
                        'cls': cls_ids[keep]
                    })
                else:
                    # No detections after confidence filtering
                    pseudo_dets.append({
                        'boxes': torch.zeros((0, 4), device=self.device),
                        'confs': torch.zeros(0, device=self.device),
                        'cls': torch.zeros(0, dtype=torch.long, device=self.device)
                    })
            else:
                # No detections at all
                pseudo_dets.append({
                    'boxes': torch.zeros((0, 4), device=self.device),
                    'confs': torch.zeros(0, device=self.device),
                    'cls': torch.zeros(0, dtype=torch.long, device=self.device)
                })
                
        return pseudo_dets
    
    def calculate_region_confidence(self, pseudo_dets):
        """Calculate confidence scores for each region"""
        regions = {'lt': [], 'rt': [], 'lb': [], 'rb': []}
        h, w = CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']
        
        for det in pseudo_dets:
            boxes = det['boxes']
            confs = det['confs']
            
            for box, conf in zip(boxes, confs):
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                
                # Determine region
                if x_center < w/2:
                    if y_center < h/2:
                        regions['lt'].append(conf.item())
                    else:
                        regions['lb'].append(conf.item())
                else:
                    if y_center < h/2:
                        regions['rt'].append(conf.item())
                    else:
                        regions['rb'].append(conf.item())
        
        # Calculate average confidence for each region
        region_confs = {}
        for region, confs in regions.items():
            if confs:
                region_confs[region] = sum(confs) / len(confs)
            else:
                region_confs[region] = 0.0
        
        # Find best region
        best_region = max(region_confs.items(), key=lambda x: x[1])
        
        return {
            'region_confs': region_confs,
            'best_region': best_region[0],
            'max_conf': best_region[1]
        }

    def create_confmix_batch(self, source_imgs, target_imgs, best_region):
        """Create ConfMix images based on selected region"""
        h, w = source_imgs.shape[-2:]
        mask = torch.zeros_like(source_imgs, device=self.device)
        
        # Set mask for selected region
        if best_region == 'lb':
            mask[:, :, h//2:, :w//2] = 1
        elif best_region == 'lt':
            mask[:, :, :h//2, :w//2] = 1
        elif best_region == 'rb':
            mask[:, :, h//2:, w//2:] = 1
        else:  # rt
            mask[:, :, :h//2, w//2:] = 1
        
        # Create mixed images
        mixed_batch = source_imgs * (1-mask) + target_imgs * mask
        return mixed_batch, mask

    def compute_combined_loss(self, source_pred, confmix_pred, source_labels, confmix_mask, gamma):
        """Compute combined loss including ConfMix loss"""
        # Create batch dictionary for YOLO loss
        batch = {
            'batch_idx': torch.zeros(len(source_labels), device=self.device),  # batch indices
            'cls': torch.cat([label[:, 0:1] for label in source_labels if len(label)]),  # class labels
            'bboxes': torch.cat([label[:, 1:] for label in source_labels if len(label)]),  # bbox coordinates
            'img_size': torch.tensor([CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']], device=self.device)  # image size
        }
        
        # Source domain supervised loss
        supervised_loss = self.model.model.loss(source_pred, batch)[0]  # Use loss method instead of criterion
        
        # ConfMix consistency loss - compare features instead of raw predictions
        if isinstance(source_pred, tuple):
            source_features = source_pred[0]  # Use first element if tuple
        else:
            source_features = source_pred
            
        if isinstance(confmix_pred, tuple):
            confmix_features = confmix_pred[0]
        else:
            confmix_features = confmix_pred
        
        # Apply mask to feature maps
        mask_expanded = confmix_mask.unsqueeze(1).expand_as(source_features)
        consistency_loss = F.mse_loss(
            confmix_features * (1 - mask_expanded),
            source_features * (1 - mask_expanded)
        )
        
        # Progressive adaptation parameter
        r = self.current_iteration / self.total_iterations
        delta = 2 / (1 + math.exp(-5 * r)) - 1
        
        # Combined loss
        total_loss = supervised_loss + delta * gamma * consistency_loss
        
        return total_loss, supervised_loss.item(), consistency_loss.item()

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.model.train()
        epoch_stats = {
            'loss': 0,
            'supervised_loss': 0,
            'consistency_loss': 0
        }
        
        target_loader_iter = iter(self.target_loader)
        
        for batch_idx, source_batch in enumerate(self.source_loader):
            try:
                target_batch = next(target_loader_iter)
            except StopIteration:
                target_loader_iter = iter(self.target_loader)
                target_batch = next(target_loader_iter)
            
            source_images = source_batch['image'].to(self.device)
            target_images = target_batch['image'].to(self.device)
            source_labels = source_batch['labels']  # List of label tensors
            
            # Forward passes with torch.amp.autocast
            with torch.amp.autocast('cuda'):
                # Source domain forward pass
                source_pred = self.model.model(source_images)
                
                # Target domain predictions for ConfMix
                with torch.no_grad():
                    target_pred = self.model.model(target_images)
                
                # Get pseudo labels and create ConfMix
                pseudo_dets = self.get_pseudo_detections(target_pred)
                region_conf = self.calculate_region_confidence(pseudo_dets)
                
                # Create ConfMix batch
                mixed_batch, mix_mask = self.create_confmix_batch(
                    source_images, target_images, region_conf['best_region']
                )
                
                # Forward pass on mixed batch
                confmix_pred = self.model.model(mixed_batch)
                
                # Calculate gamma
                gamma = float(region_conf['max_conf'] > CONFIG['GAMMA_THRESHOLD'])
                
                # Compute losses
                loss, sup_loss, cons_loss = self.compute_combined_loss(
                    source_pred, confmix_pred, source_labels, mix_mask, gamma
                )
            
            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            epoch_stats['loss'] += loss.item()
            epoch_stats['supervised_loss'] += sup_loss
            epoch_stats['consistency_loss'] += cons_loss
            
            self.current_iteration += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{CONFIG['EPOCHS']} "
                    f"[{batch_idx}/{len(self.source_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(Sup: {sup_loss:.4f}, Cons: {cons_loss:.4f}) "
                    f"Gamma: {gamma:.2f}"
                )
        
        # Calculate epoch averages
        num_batches = len(self.source_loader)
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
        
        return epoch_stats

    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting ConfMix training for {CONFIG['EPOCHS']} epochs")
        
        for epoch in range(CONFIG['EPOCHS']):
            # Train epoch
            epoch_stats = self.train_epoch(epoch)
            
            # Log epoch statistics
            self.logger.info(
                f"Epoch {epoch+1}/{CONFIG['EPOCHS']} completed. "
                f"Average Loss: {epoch_stats['loss']:.4f} "
                f"(Sup: {epoch_stats['supervised_loss']:.4f}, "
                f"Cons: {epoch_stats['consistency_loss']:.4f})"
            )
            
            # Save checkpoint if improved
            if epoch_stats['loss'] < self.best_loss:
                self.best_loss = epoch_stats['loss']
                save_path = Path(PATHS['OUTPUT_DIR']) / "best_model.pt"
                torch.save(self.model.model.state_dict(), str(save_path))
                self.logger.info(f"New best model saved to {save_path}")
            
            # Regular checkpoint saving
            if (epoch + 1) % 5 == 0:
                save_path = Path(PATHS['OUTPUT_DIR']) / f"epoch_{epoch+1}.pt"
                torch.save(self.model.model.state_dict(), str(save_path))
                self.logger.info(f"Checkpoint saved to {save_path}")


def main():
    # Set random seeds
    random.seed(CONFIG['SEED'])
    np.random.seed(CONFIG['SEED'])
    torch.manual_seed(CONFIG['SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['SEED'])
        torch.cuda.manual_seed_all(CONFIG['SEED'])
    torch.backends.cudnn.deterministic = True
    
    # Create output directory
    os.makedirs(PATHS['OUTPUT_DIR'], exist_ok=True)
    
    # Initialize and start training
    trainer = ConfMixTrainer(
        model_path=PATHS['YOLO_MODEL'],
        source_path=PATHS['SOURCE_DATA'],
        target_path=PATHS['TARGET_DATA']
    )
    
    try:
        trainer.train()
    except Exception as e:
        trainer.logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()