# src/active_learning/config.py
import os

def get_active_learning_config_path():
    """Returns the default path to the active learning config file."""
    # Get the directory where this config.py file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Build path to active learning config
    return os.path.join(project_root, 'config', 'active_learning', 'instrument_config.yaml')