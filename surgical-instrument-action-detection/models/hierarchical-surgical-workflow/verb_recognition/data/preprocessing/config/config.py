CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.3
OUTPUT_SIZE = (256, 256)

# List of videos to process
VIDEOS_TO_PROCESS = [
    "VID01", "VID02", "VID04", "VID05", "VID06", "VID08", "VID10",
    "VID12", "VID13", "VID14", "VID15", "VID18", "VID22", "VID23",
    "VID25", "VID26", "VID27", "VID29", "VID31", "VID32", "VID35",
    "VID36", "VID40", "VID42", "VID43", "VID47", "VID48", "VID49",
    "VID50", "VID51", "VID52", "VID56", "VID57", "VID60", "VID62",
    "VID65", "VID66", "VID68", "VID70", "VID73", "VID74", "VID75",
    "VID78", "VID79", "VID80"
]

# Mappings
INSTRUMENT_MAPPING = {
    0: "grasper", 1: "bipolar", 2: "hook",
    3: "scissors", 4: "clipper", 5: "irrigator"
}

VERB_MAPPING = {
    0: 'grasp', 1: 'retract', 2: 'dissect', 3: 'coagulate',
    4: 'clip', 5: 'cut', 6: 'aspirate', 7: 'irrigate',
    8: 'pack', 9: 'null_verb'
}