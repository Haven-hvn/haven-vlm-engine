from typing import Dict, Any

from lib.config.config_utils import load_config


post_processing_config: Dict[str, Any] = load_config("./config/post_processing/post_processing_config.yaml")

def get_or_default(data_dict: Dict[str, Any], key: str, default_value: Any) -> Any:
    if key in data_dict and data_dict[key]:
        return data_dict[key]
    else:
        csv_defaults: Dict[str, Any] = post_processing_config.get("csv_defaults", {})
        return csv_defaults.get(key, default_value)