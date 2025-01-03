import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from collections import Counter

def get_verb_instrument(filename):
    """Extract verb and instrument from filename"""
    base = filename.rsplit('.', 1)[0]
    parts = base.split('_')
    
    instrument = parts[1]
    if len(parts) == 5 and parts[2] == "null" and parts[3] == "verb":
        verb = "null_verb"
    else:
        verb = parts[2]
    
    return instrument, verb

def analyze_distribution(base_dir):
    # Verbindliche Listen für Verben und Instrumente
    verb_names = ["dissect", "retract", "null_verb", "coagulate", "grasp",
                 "clip", "aspirate", "cut", "irrigate"]
    instrument_names = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
    
    # Matrix der gültigen Kombinationen
    instrument_verb_matrix = np.array([
        [0, 0, 1, 1, 0, 0],  # dissect
        [1, 1, 1, 0, 0, 1],  # retract
        [1, 1, 1, 1, 1, 1],  # null_verb
        [0, 1, 1, 0, 0, 0],  # coagulate
        [1, 1, 0, 0, 0, 0],  # grasp
        [0, 0, 0, 0, 1, 0],  # clip
        [0, 0, 0, 0, 0, 1],  # aspirate
        [0, 0, 0, 1, 0, 0],  # cut
        [0, 0, 0, 0, 0, 1],  # irrigate
    ])
    
    # Sammle Paare
    pairs = []
    for vid_dir in os.listdir(base_dir):
        if not vid_dir.startswith('VID'):
            continue
            
        vid_path = os.path.join(base_dir, vid_dir)
        for filename in os.listdir(vid_path):
            if not filename.endswith('.png'):
                continue
                
            instrument, verb = get_verb_instrument(filename)
            if instrument in instrument_names and verb in verb_names:
                pairs.append({
                    'instrument': instrument,
                    'verb': verb
                })
    
    # Erstelle DataFrame und zähle Paare
    df = pd.DataFrame(pairs)
    
    # Debug: Prüfe ob Daten vorhanden sind
    print(f"Found {len(pairs)} valid pairs")
    if len(pairs) > 0:
        print("\nInstrument distribution:")
        print(df['instrument'].value_counts())
        print("\nVerb distribution:")
        print(df['verb'].value_counts())
    
    # Erstelle Pivot-Tabelle
    pivot_table = pd.crosstab(df['instrument'], df['verb'])
    
    # Fülle fehlende Kombinationen mit 0
    for verb in verb_names:
        if verb not in pivot_table.columns:
            pivot_table[verb] = 0
    for inst in instrument_names:
        if inst not in pivot_table.index:
            pivot_table.loc[inst] = 0
    
    # Sortiere nach vordefinierten Listen
    pivot_table = pivot_table.reindex(index=instrument_names, columns=verb_names)
    pivot_table = pivot_table.fillna(0)  # Konvertiere NaN zu 0
    
    # Debug: Zeige Pivot-Tabelle
    print("\nPivot table:")
    print(pivot_table)
    
    # Erstelle Plot
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
    
    plt.title('Distribution of Valid Instrument-Verb Pairs')
    plt.xlabel('Verbs')
    plt.ylabel('Instruments')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('verb_instrument_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    base_dir = "/data/Bartscht/Verbs"
    analyze_distribution(base_dir)