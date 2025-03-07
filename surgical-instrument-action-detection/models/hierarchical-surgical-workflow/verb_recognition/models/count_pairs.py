#!/usr/bin/env python3
import os
import re
from collections import Counter
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def count_instrument_verb_pairs(base_dir):
    """
    Zählt die Häufigkeit aller Instrument-Verb-Paare in den Dateinamen
    innerhalb der angegebenen Verzeichnisse.
    
    Args:
        base_dir: Basis-Verzeichnis, das die VID* Unterordner enthält
    
    Returns:
        Counter-Objekt mit Häufigkeiten der Instrument-Verb-Paare
    """
    # Regulärer Ausdruck für Dateinamen im Format: "{frame_id}_{instrument}_{verb}_conf{confidence}.png"
    pattern = r'(\d+)_(\w+)_(\w+)_conf([\d\.]+)\.png'
    
    # Counter für die Instrument-Verb-Paare
    pair_counter = Counter()
    
    # Alle VID* Verzeichnisse durchsuchen
    vid_dirs = [d for d in os.listdir(base_dir) if d.startswith('VID') and os.path.isdir(os.path.join(base_dir, d))]
    
    # Gesamtzahl der Dateien für Fortschrittsanzeige
    total_files = 0
    processed_files = 0
    
    # Zuerst die Gesamtzahl der Dateien bestimmen
    print("Zähle Dateien...")
    for vid_dir in vid_dirs:
        vid_path = os.path.join(base_dir, vid_dir)
        for root, _, files in os.walk(vid_path):
            png_files = [f for f in files if f.endswith('.png')]
            total_files += len(png_files)
    
    print(f"Insgesamt {total_files} Dateien gefunden.")
    
    # Jetzt die Dateien verarbeiten
    for vid_dir in vid_dirs:
        vid_path = os.path.join(base_dir, vid_dir)
        print(f"Verarbeite Verzeichnis: {vid_dir}")
        
        for root, _, files in os.walk(vid_path):
            png_files = [f for f in files if f.endswith('.png')]
            
            for filename in png_files:
                match = re.match(pattern, filename)
                if match:
                    _, instrument, verb, _ = match.groups()
                    
                    # Erste Buchstabe groß schreiben, außer bei 'null_verb'
                    instrument = instrument.capitalize()
                    if verb != 'null_verb':
                        verb = verb.capitalize()
                    
                    # Instrument-Verb-Paar im Format "Instrument-Verb"
                    pair = f"{instrument}-{verb}"
                    pair_counter[pair] += 1
                
                # Fortschritt anzeigen
                processed_files += 1
                if processed_files % 1000 == 0:
                    print(f"Verarbeitet: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)")
    
    return pair_counter

def save_results(pair_counter, output_file):
    """
    Speichert die Ergebnisse in einer Textdatei.
    
    Args:
        pair_counter: Counter-Objekt mit Instrument-Verb-Paaren
        output_file: Pfad zur Ausgabedatei
    """
    # Nach Häufigkeit sortieren (absteigend)
    sorted_pairs = pair_counter.most_common()
    
    with open(output_file, 'w') as f:
        for pair, count in sorted_pairs:
            f.write(f"{pair} {count}\n")
    
    print(f"Ergebnisse wurden in {output_file} gespeichert.")

def plot_distribution(pair_counter, output_file=None):
    """
    Erstellt eine Visualisierung der Verteilung der Instrument-Verb-Paare.
    
    Args:
        pair_counter: Counter-Objekt mit Instrument-Verb-Paaren
        output_file: Pfad zum Speichern der Visualisierung (optional)
    """
    # Nach Häufigkeit sortieren (absteigend)
    sorted_pairs = pair_counter.most_common()
    
    # In DataFrame konvertieren für einfachere Visualisierung
    df = pd.DataFrame(sorted_pairs, columns=['Paar', 'Anzahl'])
    
    # Zu viele Paare können unübersichtlich sein, daher die Top-N anzeigen
    if len(df) > 30:
        top_df = df.head(25)
        other_count = df.iloc[25:]['Anzahl'].sum()
        top_df = pd.concat([top_df, pd.DataFrame([['Andere', other_count]], columns=['Paar', 'Anzahl'])])
        df = top_df
    
    # Balkendiagramm erstellen
    plt.figure(figsize=(12, 10))
    bars = plt.barh(df['Paar'], df['Anzahl'])
    
    # Werte am Ende der Balken anzeigen
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', ha='left', va='center')
    
    plt.xlabel('Anzahl')
    plt.ylabel('Instrument-Verb-Paar')
    plt.title('Häufigkeitsverteilung der Instrument-Verb-Paare')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualisierung wurde in {output_file} gespeichert.")
    
    plt.show()

def calculate_class_weights(pair_counter, method='inverse'):
    """
    Berechnet Klassengewichtungen basierend auf den Häufigkeiten.
    
    Args:
        pair_counter: Counter-Objekt mit Instrument-Verb-Paaren
        method: Methode zur Berechnung der Gewichtungen ('inverse', 'sqrt_inverse', oder 'effective_samples')
        
    Returns:
        Dictionary mit Gewichtungen pro Klasse
    """
    import numpy as np
    
    total = sum(pair_counter.values())
    class_weights = {}
    
    if method == 'inverse':
        # Inverse Frequency Weighting
        for pair, count in pair_counter.items():
            class_weights[pair] = total / (count * len(pair_counter))
    
    elif method == 'sqrt_inverse':
        # Square Root Inverse Frequency Weighting
        for pair, count in pair_counter.items():
            class_weights[pair] = np.sqrt(total / (count * len(pair_counter)))
    
    elif method == 'effective_samples':
        # Effective Number of Samples Weighting
        beta = 0.9999
        for pair, count in pair_counter.items():
            effective_num = (1 - beta**count) / (1 - beta)
            class_weights[pair] = 1 / effective_num
        
        # Normalisieren
        max_weight = max(class_weights.values())
        for pair in class_weights:
            class_weights[pair] /= max_weight
            class_weights[pair] *= 10  # Skalierungsfaktor
    
    return class_weights

def main():
    parser = argparse.ArgumentParser(description='Zählt Instrument-Verb-Paare in Dateinamen.')
    parser.add_argument('base_dir', help='Basisverzeichnis mit VID* Ordnern')
    parser.add_argument('--output', '-o', help='Ausgabedatei für die Ergebnisse', default='instrument_verb_pairs.txt')
    parser.add_argument('--plot', '-p', help='Visualisierungsdatei (z.B. plot.png)', default='pair_distribution.png')
    parser.add_argument('--weights', '-w', choices=['inverse', 'sqrt_inverse', 'effective_samples'],
                        help='Methode zur Berechnung der Klassengewichtungen', default='inverse')
    
    args = parser.parse_args()
    
    # Prüfen, ob das Basisverzeichnis existiert
    if not os.path.isdir(args.base_dir):
        print(f"Fehler: Verzeichnis {args.base_dir} existiert nicht.")
        return
    
    # Instrument-Verb-Paare zählen
    pair_counter = count_instrument_verb_pairs(args.base_dir)
    
    # Ergebnisse speichern
    save_results(pair_counter, args.output)
    
    # Klassengewichtungen berechnen
    weights = calculate_class_weights(pair_counter, method=args.weights)
    
    # Gewichtungen speichern
    weights_file = os.path.splitext(args.output)[0] + '_weights.txt'
    with open(weights_file, 'w') as f:
        f.write(f"# Klassengewichtungen (Methode: {args.weights})\n")
        for pair, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{pair} {weight:.2f}\n")
    
    print(f"Klassengewichtungen wurden in {weights_file} gespeichert.")
    
    # Verteilung visualisieren
    plot_distribution(pair_counter, args.plot)
    
    # Statistiken anzeigen
    total_samples = sum(pair_counter.values())
    num_classes = len(pair_counter)
    most_common = pair_counter.most_common(1)[0]
    least_common = pair_counter.most_common()[-1]
    
    print("\nZusammenfassung:")
    print(f"Gesamtzahl der Samples: {total_samples}")
    print(f"Anzahl der verschiedenen Instrument-Verb-Paare: {num_classes}")
    print(f"Häufigstes Paar: {most_common[0]} ({most_common[1]} Vorkommen)")
    print(f"Seltenestes Paar: {least_common[0]} ({least_common[1]} Vorkommen)")
    print(f"Ungleichgewichtsverhältnis: {most_common[1]/least_common[1]:.2f}:1")

if __name__ == "__main__":
    main()