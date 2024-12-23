from .core.core_processing import process_all_videos
from .config.config import (
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    OUTPUT_SIZE,
    VIDEOS_TO_PROCESS,
    INSTRUMENT_MAPPING,
    VERB_MAPPING
)

__all__ = [
    'process_all_videos',
    'CONFIDENCE_THRESHOLD',
    'IOU_THRESHOLD',
    'OUTPUT_SIZE',
    'VIDEOS_TO_PROCESS',
    'INSTRUMENT_MAPPING',
    'VERB_MAPPING'
]