import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Verzeichnis mit CSV-Dateien
csv_dir = '/data/Bartscht/cropped_images/labels'

# Liste aller möglichen Verben (kann angepasst werden basierend auf Ihren Daten)
verb_classes = ["grasp", "retract", "dissect", "coagulate", "clip", "cut", "aspirate", "irrigate", "pack"]

# Funktion zum Lesen und Verarbeiten einer CSV-Datei
def process_csv(file_path):
    df = pd.read_csv(file_path)
    # Extrahiere Instrument aus Dateinamen (zweites Element nach Split)
    df['Instrument'] = df['Dateiname'].str.split('_').str[1]
    return df

# Alle CSV-Dateien einlesen und verarbeiten
all_data = []
for file in os.listdir(csv_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(csv_dir, file)
        try:
            df = process_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {file}: {e}")

# Alle Daten zusammenführen
combined_data = pd.concat(all_data, ignore_index=True)

# Häufigkeitsverteilung der Verbklassen
verb_counts = combined_data['Verb'].value_counts()

# Plot für Verbklassen-Häufigkeit mit absoluten Zahlen über den Balken
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=verb_counts.index, y=verb_counts.values)
plt.xlabel('Verb')
plt.ylabel('Häufigkeit')
plt.xticks(rotation=45)

# Hinzufügen der absoluten Zahlen über den Balken
for i, v in enumerate(verb_counts.values):
    ax.text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('verb_distribution.png')
plt.close()

# Häufigkeitsverteilung von [Instrument]+[Verb]
instrument_verb_counts = combined_data.groupby(['Instrument', 'Verb']).size().reset_index(name='count')
instrument_verb_counts = instrument_verb_counts.sort_values('count', ascending=False)

# Plot für [Instrument]+[Verb]-Häufigkeit
plt.figure(figsize=(15, 8))
sns.barplot(x='Instrument', y='count', hue='Verb', data=instrument_verb_counts)
plt.xlabel('Instrument')
plt.ylabel('Häufigkeit')
plt.xticks(rotation=45)
plt.legend(title='Verb', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('instrument_verb_distribution.png')
plt.close()

# Tabellenoutput für Verbklassen-Häufigkeit
verb_counts_table = verb_counts.reset_index()
verb_counts_table.columns = ['Verb', 'Häufigkeit']
verb_counts_table.to_csv('verb_counts.csv', index=False)
print("Verbklassen-Häufigkeit als CSV gespeichert: verb_counts.csv")

# Tabellenoutput für [Instrument]+[Verb]-Häufigkeit
instrument_verb_counts.to_csv('instrument_verb_counts.csv', index=False)
print("Instrument+Verb-Häufigkeit als CSV gespeichert: instrument_verb_counts.csv")

print("Analyse abgeschlossen. Die Plots wurden als 'verb_distribution.png' und 'instrument_verb_distribution.png' gespeichert.")
print("Die Häufigkeitstabellen wurden als 'verb_counts.csv' und 'instrument_verb_counts.csv' gespeichert.")