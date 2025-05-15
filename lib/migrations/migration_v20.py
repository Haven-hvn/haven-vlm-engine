import os
import yaml
from typing import Dict, Any, List, Optional, Union, TextIO

model_directory: str = "./config/models"

# Define a more specific type for the values in model_fields_to_add if possible
# For now, Any is used, but could be Dict[str, Union[str, int, float, List[str]]] for example
ModelFieldDetail = Dict[str, Any]

model_fields_to_add: Dict[str, ModelFieldDetail] = {
    "gentler_river": {"model_category": ["actions"], "model_version": 2.0, "model_image_size": 512, "model_info": "Most Accurate", "model_identifier": 200},
    "gentler_river_full": {"model_category": ["actions"], "model_version": 2.0, "model_image_size": 512, "model_info": "(CPU Variant if no Nvidia GPU available) Most Accurate", "model_identifier": 900},
    "vivid_galaxy": {"model_category": ["actions"], "model_version": 1.9, "model_image_size": 512, "model_info": "Free Variant", "model_identifier": 950, "normalization_config": 0},
    "vivid_galaxy_full": {"model_category": ["actions"], "model_version": 1.9, "model_image_size": 512, "model_info": "(CPU Variant if no Nvidia GPU available) Free Variant", "model_identifier": 970, "normalization_config": 0},
    "distinctive_haze": {"model_category": ["actions"], "model_version": 2.0, "model_image_size": 384, "model_info": "Faster but slightly less accurate", "model_identifier": 400},
    "iconic_sky": {"model_category": ["bodyparts"], "model_version": 0.5, "model_image_size": 512, "model_info": "Most Accurate", "model_identifier": 200},
    "true_lake": {"model_category": ["bdsm"], "model_version": 0.7, "model_image_size": 512, "model_info": "Most Accurate", "model_identifier": 200},
}

def migrate_to_2_0() -> None:
    root: str
    files: List[str]
    # _: List[str] # for dirs, if not used
    for root, _, files in os.walk(model_directory):
        file_loop_var: str # Renamed file to avoid conflict
        for file_loop_var in files:
            if file_loop_var.endswith('.yaml'):
                yaml_path: str = os.path.join(root, file_loop_var)
                f_read: TextIO
                with open(yaml_path, 'r') as f_read:
                    data: Optional[Dict[str, Any]] = yaml.safe_load(f_read)

                if data and data.get('type') == 'model':
                    model_file_name: Optional[str] = data.get('model_file_name')
                    if model_file_name and model_file_name in model_fields_to_add:
                        fields_to_add: ModelFieldDetail = model_fields_to_add[model_file_name]

                        updated: bool = False
                        key: str
                        value: Any
                        for key, value in fields_to_add.items():
                            if key not in data:
                                data[key] = value
                                updated = True

                        if updated:
                            f_write: TextIO
                            with open(yaml_path, 'w') as f_write:
                                yaml.dump(data, f_write, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    migrate_to_2_0()