import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append('/home/Bartscht/YOLO')
from Verbmodel.utils.finaltest import VerbDataModule
import numpy as np 
import torch
from torch.utils.data import Subset
from torchvision import transforms

# Basis-Verzeichnisse
base_dir = "/data/Bartscht/Verbs"
labels_dir = os.path.join(base_dir, "labels")

# Spezifizierte Instrument-Verb-Kombinationen
specific_combinations = [
    ("hook", "dissect"),
    ("grasper", "retract"),
    ("bipolar", "coagulate"),
    ("grasper", "grasp"),
    ("hook", "null_verb"),
    ("clipper", "clip"),
    ("irrigator", "aspirate"),
    ("grasper", "null_verb"),
    ("scissors", "cut")
]

def load_all_data():
    all_data = []
    for csv_file in os.listdir(labels_dir):
        if csv_file.endswith(".csv"):
            vid = csv_file.split(".")[0]  # VID01 aus VID01.csv
            try:
                df = pd.read_csv(os.path.join(labels_dir, csv_file))
                df['VID'] = vid
                all_data.append(df)
            except Exception as e:
                print(f"Fehler beim Lesen von {csv_file}: {e}")
    return pd.concat(all_data, ignore_index=True)

def get_instrument(filename):
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""

def select_images(df):
    selected_images = []
    for instrument, verb in specific_combinations:
        matching_df = df[(df['Dateiname'].apply(get_instrument) == instrument) & (df['Verb'] == verb)]
        if not matching_df.empty:
            selected_image = matching_df.sample(n=1).iloc[0]
            selected_images.append(selected_image)
        else:
            print(f"Keine Übereinstimmung gefunden für: {instrument} - {verb}")
    return selected_images

def visualize_images(images):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    #plt.suptitle('Figure 1: Surgical Instrument Actions', fontsize=24, y=1.02)
    
    for i, img_data in enumerate(images):
        if i >= len(axes):
            break
            
        vid_folder = img_data['VID']
        img_path = os.path.join(base_dir, vid_folder, img_data['Dateiname'])
        
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Paper-style subtitles with letters
            instrument = get_instrument(img_data['Dateiname']).capitalize()
            verb = img_data['Verb'].lower()
            subtitle = f"({chr(97+i)}) {instrument} {verb}"
            axes[i].set_title(subtitle, fontsize=16, pad=10)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[i].axis('off')
    
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('surgical_instruments_visualization.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Visualization saved as 'surgical_instruments_visualization.png'")
def visualize_specific_combinations(dataloader):
    # Anpassung der Transformationen für Validierung (keine Augmentationen)
    dataloader.dataset.dataset.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    specific_combinations = [
        ("hook", "dissect"), ("grasper", "retract"), ("bipolar", "coagulate"),
        ("grasper", "grasp"), ("hook", "null_verb"), ("clipper", "clip"),
        ("irrigator", "aspirate"), ("grasper", "null_verb"), ("scissors", "cut")
    ]
    
    found_images = []
    found_pairs = set()
    
    for _ in range(10):
        images, instruments, verb_labels = next(iter(dataloader))
        verb_names = dataloader.dataset.dataset.verb_names if isinstance(dataloader.dataset, Subset) else dataloader.dataset.verb_names
        
        for img, inst, verb_idx in zip(images, instruments, verb_labels):
            pair = (inst, verb_names[verb_idx])
            if pair in specific_combinations and pair not in found_pairs:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img.cpu().unsqueeze(0)
                img = img * std + mean
                img = img.squeeze(0).permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                
                found_images.append((img, pair))
                found_pairs.add(pair)
                
                if len(found_images) == 9:
                    break
        
        if len(found_images) == 9:
            break
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    #plt.suptitle('Figure 1: Surgical Instrument Actions', fontsize=24, y=1.02)
    
    for i, (target_pair) in enumerate(specific_combinations[:9]):
        ax = axes[i//3, i%3]
        
        matching_img = None
        for img, pair in found_images:
            if pair == target_pair:
                matching_img = img
                break
        
        if matching_img is not None:
            ax.imshow(matching_img)
            ax.axis('off')
            
            title = f"({chr(97+i)}) {target_pair[0]} {target_pair[1]}"
            ax.set_title(title, fontsize=16, pad=10, y=-0.1, weight='bold')
    
    plt.tight_layout()
    plt.savefig('specific_combinations_visualization.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    base_dir = "/data/Bartscht/Verbs"
    data_module = VerbDataModule(base_dir)
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    visualize_specific_combinations(train_loader)
    print("Visualization saved as 'specific_combinations_visualization.png'")