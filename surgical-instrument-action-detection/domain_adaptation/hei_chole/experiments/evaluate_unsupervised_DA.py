import torch
from pathlib import Path
from dataloader import balanced_dataloader  
from unsupervised_DA_instruments import DomainAdapter    

def load_feature_reducer(model, checkpoint_path):
    """Load the saved feature reducer weights into the model"""
    checkpoint = torch.load(checkpoint_path)
    model.feature_reducer.load_state_dict(checkpoint['feature_reducer'])
    return model

def evaluate_features(model, test_loader, device):
    """Evaluate the feature reducer on the test set"""
    model.eval()
    domain_predictions = []
    true_domains = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            domains = batch['domain'].to(device)
            
            # Get domain predictions
            domain_pred = model(images, alpha=1.0)
            domain_preds = (domain_pred.squeeze() > 0.5).float()
            
            domain_predictions.extend(domain_preds.cpu().numpy())
            true_domains.extend(domains.cpu().numpy())
    
    # Calculate metrics
    domain_predictions = torch.tensor(domain_predictions)
    true_domains = torch.tensor(true_domains)
    
    accuracy = (domain_predictions == true_domains).float().mean().item()
    
    # Calculate separate accuracies for source and target domains
    source_mask = true_domains == 0
    target_mask = true_domains == 1
    
    source_acc = (domain_predictions[source_mask] == true_domains[source_mask]).float().mean().item() if source_mask.any() else 0
    target_acc = (domain_predictions[target_mask] == true_domains[target_mask]).float().mean().item() if target_mask.any() else 0
    
    return {
        'overall_accuracy': accuracy,
        'source_accuracy': source_acc,
        'target_accuracy': target_acc,
        'domain_confusion_score': 1 - abs(2 * accuracy - 1)
    }

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path("/home/Bartscht/YOLO/surgical-instrument-action-detection/domain_adaptation/hei_chole/experiments/domain_adapter_weights/best_feature_reducer.pt")
    yolo_path = "/data/Bartscht/YOLO/best_v35.pt"  # Your YOLO path
    
    # Initialize model and load weights
    model = DomainAdapter(yolo_path=yolo_path).to(device)
    model = load_feature_reducer(model, checkpoint_path)
    
    # Get test dataloader (you might need to modify this based on your dataloader)
    test_loader = balanced_dataloader(split='test')
    
    # Evaluate
    print("\nEvaluating feature reducer on test set...")
    metrics = evaluate_features(model, test_loader, device)
    
    # Print results
    print("\nTest Set Metrics:")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Source Domain Accuracy: {metrics['source_accuracy']:.4f}")
    print(f"Target Domain Accuracy: {metrics['target_accuracy']:.4f}")
    print(f"Domain Confusion Score: {metrics['domain_confusion_score']:.4f}")

if __name__ == "__main__":
    main()