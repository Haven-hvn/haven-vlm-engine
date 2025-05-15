import csv
import os
from typing import Dict, List, TextIO

category_config: Dict[str, Dict[str, Dict[str, str]]] = {}

category_directory: str = "./config/post_processing/categories"

def load_category_settings() -> None:
    root: str
    # dirs: List[str] # If needed, but _ is used
    files: List[str]
    for root, _, files in os.walk(category_directory):
        file: str
        for file in files:
            if file.endswith('.csv'):
                csv_path: str = os.path.join(root, file)
                file_name: str = os.path.splitext(file)[0]
                if file_name not in category_config:
                    category_config[file_name] = {}  # Initialize as an empty dictionary
                
                f: TextIO
                with open(csv_path, 'r') as f:
                    reader: csv.DictReader = csv.DictReader(f)
                    row: Dict[str, str]
                    for row in reader:
                        category_config[file_name][row['OriginalTag']] = {
                            'RenamedTag': row['RenamedTag'],
                            'MinMarkerDuration': row['MinMarkerDuration'],
                            'MaxGap': row['MaxGap'],
                            'RequiredDuration': row['RequiredDuration'],
                            'TagThreshold': row['TagThreshold'],
                        }
    
load_category_settings()