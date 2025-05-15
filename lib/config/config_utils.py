import yaml
import os
from typing import Dict, Any, Optional, TextIO

def load_config(file_path: str, default_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Load YAML configuration from the specified file path.
    Returns a dictionary with configuration values.
    """
    if default_config is None:
        default_config = {}

    # Check if the file exists
    if not os.path.exists(file_path):
        if default_config is not None: #This condition was previously default_config is None which is incorrect for returning default_config
            print(f"WARNING: Configuration file {file_path} not found. Using default values provided.")
            return default_config
        else:
            print(f"ERROR: Configuration file {file_path} not found and no defaults provided. Please ensure this config file is present!!")
            return None # No file and no defaults, return None

    try:
        # Open the YAML file and load it
        f: TextIO
        with open(file_path, 'r') as f:
            config: Optional[Dict[str, Any]] = yaml.safe_load(f)
        
        # Update the default configuration with values from the file
        # This ensures that any missing keys in the YAML file will use the defaults
        # Config can be None if the YAML file is empty.
        updated_config: Dict[str, Any] = {**default_config, **(config if config is not None else {})}
        return updated_config
    except yaml.YAMLError as e_yaml: # Renamed e
        print(f"Error loading the YAML file {file_path}: {e_yaml}")
        # Return defaults if YAML is malformed and defaults are available
        return default_config 
    except Exception as e_general: # Renamed e
        print(f"An unexpected error occurred while reading {file_path}: {e_general}")
        # Return defaults on other errors if defaults are available
        return default_config