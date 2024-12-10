import os
import json

def convert_json_to_txt(dataset_dir, output_file):
    labels_folder = os.path.join(dataset_dir, "labels")
    test_videos = ['VID92', 'VID96', 'VID103', 'VID110', 'VID111']
    
    instrument_names = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen bag"]
    verb_names = ["grasp", "retract", "dissect", "coagulate", "clip", "cut", "aspirate", "irrigate", "pack", "null_verb"]
    
    with open(output_file, 'w') as out_file:
        for video in test_videos:
            json_file = os.path.join(labels_folder, f"{video}.json")
            
            with open(json_file, 'r') as f:
                data = json.load(f)
                annotations = data['annotations']
                
                out_file.write(f"Video: {video}\n")
                out_file.write("=" * 50 + "\n")
                
                for frame, instances in annotations.items():
                    valid_instances = []
                    
                    for instance in instances:
                        instrument = instance[1]
                        verb = instance[7]
                        
                        # Check if the instance is valid (not unknown)
                        if (isinstance(instrument, int) and 0 <= instrument < len(instrument_names)) or \
                           (isinstance(instrument, list) and any(instrument)):
                            if (isinstance(verb, int) and 0 <= verb < len(verb_names)) or \
                               (isinstance(verb, list) and any(verb)):
                                valid_instances.append((instrument, verb))
                    
                    # Only write frame information if there are valid instances
                    if valid_instances:
                        out_file.write(f"Frame: {frame}\n")
                        
                        for instrument, verb in valid_instances:
                            # Process instrument
                            if isinstance(instrument, int):
                                instrument_str = instrument_names[instrument]
                            elif isinstance(instrument, list):
                                present_instruments = [instrument_names[i] for i, v in enumerate(instrument) if v == 1 and i < len(instrument_names)]
                                instrument_str = ', '.join(present_instruments)
                            
                            # Process verb
                            if isinstance(verb, int):
                                verb_str = verb_names[verb]
                            elif isinstance(verb, list):
                                present_verbs = [verb_names[i] for i, v in enumerate(verb) if v == 1 and i < len(verb_names)]
                                verb_str = ', '.join(present_verbs)
                            
                            # Write combined information
                            out_file.write(f"  Instrument: {instrument_str} | Verb: {verb_str}\n")
                        
                        out_file.write("-" * 30 + "\n")
                
                out_file.write("\n")

# Usage
dataset_dir = "/data/Bartscht/CholecT50"
output_file = "testset_IV_Labels.txt"

convert_json_to_txt(dataset_dir, output_file)
print(f"Conversion complete. Results saved to {output_file}")