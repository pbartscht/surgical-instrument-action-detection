import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions import Normal

class ConfMixCore:
    def __init__(self, model, device='cuda'):
        """
        Core implementation of ConfMix strategy
        
        Args:
            model: YOLO model
            device: torch device
        """
        self.model = model.to(device)
        self.device = device
        
        # Confidence thresholds
        self.conf_thres = 0.25  # Base confidence threshold
        self.c_gamma_thres = 0.5  # Threshold for gamma calculation
        
        # Progressive pseudo-labeling parameters
        self.alpha = 5.0  # Controls progression speed
        
        # Region division parameters
        self.num_regions = 4
        self.region_weights = {
            'confidence': 0.6,
            'density': 0.4
        }
    
    def forward_step(self, source_batch, target_batch, curr_iteration, max_iterations):
        """
        Perform one forward step of ConfMix
        
        Args:
            source_batch: Batch from source domain
            target_batch: Batch from target domain
            curr_iteration: Current training iteration
            max_iterations: Total training iterations
        """
        # Compute progressive ratio
        r = curr_iteration / max_iterations
        delta = 2 / (1 + math.exp(-5. * r)) - 1
        
        # Get source predictions
        source_imgs = source_batch[0].to(self.device)
        source_targets = source_batch[1].to(self.device)
        pseudo_s, pred_s, var_s = self.model(source_imgs, pseudo=True, delta=delta)
        
        # Get target predictions
        target_imgs = target_batch[0].to(self.device)
        pseudo_t, pred_t, var_t = self.model(target_imgs, pseudo=True, delta=delta)
        
        # Process predictions with confidence
        out_s = self._process_predictions(pseudo_s, conf_thres=self.conf_thres)
        out_t = self._process_predictions(pseudo_t, conf_thres=self.conf_thres)
        
        # Select best region and create mixing mask
        batch_size, _, h, w = source_imgs.shape
        mixing_masks = []
        mixed_images = []
        mixed_targets = []
        
        for b in range(batch_size):
            # Get single image predictions
            b_out_t = out_t[out_t[:, 0] == b]
            
            # Divide into regions and select best
            regions = self._divide_regions(b_out_t, h, w)
            best_region_idx = self._select_best_region(regions, b_out_t)
            
            # Create mixing mask
            mask = torch.zeros((h, w), device=self.device)
            x1, y1, x2, y2 = self._get_region_coords(best_region_idx, h, w)
            mask[y1:y2, x1:x2] = 1
            mixing_masks.append(mask)
            
            # Mix images
            mixed_img = (source_imgs[b] * (1 - mask) + 
                        target_imgs[b] * mask)
            mixed_images.append(mixed_img)
            
            # Combine predictions
            mixed_target = self._combine_predictions(
                out_s[out_s[:, 0] == b],
                b_out_t,
                best_region_idx,
                h, w
            )
            mixed_targets.append(mixed_target)
        
        # Stack results
        mixing_masks = torch.stack(mixing_masks)
        mixed_images = torch.stack(mixed_images)
        mixed_targets = torch.cat(mixed_targets)
        
        # Calculate mixing confidence (gamma)
        gamma = self._calculate_gamma(mixed_targets)
        
        return {
            'mixed_images': mixed_images,
            'mixed_targets': mixed_targets,
            'mixing_masks': mixing_masks,
            'gamma': gamma,
            'source_pred': pred_s,
            'source_var': var_s,
            'target_pred': pred_t,
            'target_var': var_t
        }
    
    def _process_predictions(self, predictions, conf_thres):
        """Process model predictions with NMS and confidence thresholding"""
        # Apply NMS
        nms_predictions = []
        for pred in predictions:
            # Apply confidence threshold
            conf_mask = pred[..., 4] > conf_thres
            pred = pred[conf_mask]
            
            if len(pred):
                # Apply NMS per class
                class_conf, class_pred = pred[..., 5:].max(1)
                pred = torch.cat((pred[..., :5], class_conf.unsqueeze(1), 
                                class_pred.unsqueeze(1)), 1)
                nms_predictions.append(pred)
            else:
                nms_predictions.append(torch.zeros((0, 7), device=self.device))
        
        return torch.cat(nms_predictions, 0)
    
    def _divide_regions(self, predictions, height, width):
        """Divide image into regions and assign predictions"""
        regions = []
        h_mid, w_mid = height // 2, width // 2
        
        # Define regions [x1, y1, x2, y2]
        region_coords = [
            [0, 0, w_mid, h_mid],          # top-left
            [w_mid, 0, width, h_mid],      # top-right
            [0, h_mid, w_mid, height],     # bottom-left
            [w_mid, h_mid, width, height]  # bottom-right
        ]
        
        for coords in region_coords:
            x1, y1, x2, y2 = coords
            mask = ((predictions[:, 1] >= x1) & (predictions[:, 1] < x2) &
                   (predictions[:, 2] >= y1) & (predictions[:, 2] < y2))
            regions.append(predictions[mask])
        
        return regions
    
    def _select_best_region(self, regions, predictions):
        """Select best region based on confidence and density"""
        region_scores = []
        
        for region_preds in regions:
            if len(region_preds) == 0:
                region_scores.append(0)
                continue
            
            # Confidence score
            conf_score = region_preds[:, 4].mean()
            
            # Density score (number of predictions normalized)
            density_score = len(region_preds) / len(predictions) if len(predictions) > 0 else 0
            
            # Combined score
            score = (self.region_weights['confidence'] * conf_score +
                    self.region_weights['density'] * density_score)
            
            region_scores.append(score)
        
        return torch.argmax(torch.tensor(region_scores))
    
    def _get_region_coords(self, region_idx, height, width):
        """Get coordinates for a region"""
        h_mid, w_mid = height // 2, width // 2
        
        if region_idx == 0:  # top-left
            return 0, 0, w_mid, h_mid
        elif region_idx == 1:  # top-right
            return w_mid, 0, width, h_mid
        elif region_idx == 2:  # bottom-left
            return 0, h_mid, w_mid, height
        else:  # bottom-right
            return w_mid, h_mid, width, height
    
    def _combine_predictions(self, source_preds, target_preds, region_idx, height, width):
        """Combine source and target predictions based on mixing region"""
        x1, y1, x2, y2 = self._get_region_coords(region_idx, height, width)
        
        # Get predictions in target region
        target_mask = ((target_preds[:, 1] >= x1) & (target_preds[:, 1] < x2) &
                      (target_preds[:, 2] >= y1) & (target_preds[:, 2] < y2))
        target_region_preds = target_preds[target_mask]
        
        # Get predictions in source region (inverse mask)
        source_mask = ~((source_preds[:, 1] >= x1) & (source_preds[:, 1] < x2) &
                       (source_preds[:, 2] >= y1) & (source_preds[:, 2] < y2))
        source_region_preds = source_preds[source_mask]
        
        # Combine predictions
        combined_preds = torch.cat([source_region_preds, target_region_preds])
        
        return combined_preds
    
    def _calculate_gamma(self, predictions):
        """Calculate mixing confidence weight gamma"""
        if len(predictions) == 0:
            return torch.tensor(0., device=self.device)
            
        # Count predictions above gamma threshold
        conf_mask = predictions[:, 4] > self.c_gamma_thres
        gamma = conf_mask.float().mean()
        
        return gamma