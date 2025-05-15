import logging
import time
from lib.async_lib.async_processing import ItemFuture, QueueItem
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import preprocess_image
from typing import Dict, Any, List, Optional, Union, Tuple

class ImagePreprocessorModel(Model):
    def __init__(self, configValues: Dict[str, Any]):
        super().__init__(configValues)
        self.image_size: Union[int, Tuple[int, int]] = configValues.get("image_size", 512)
        self.use_half_precision: bool = bool(configValues.get("use_half_precision", True))
        self.device: Optional[str] = configValues.get("device")
        self.normalization_config: Union[int, Dict[str, List[float]]] = configValues.get("normalization_config", 1)
        self.logger: logging.Logger = logging.getLogger("logger")
    
    async def worker_function(self, queue_items: List[QueueItem]) -> None:
        item: QueueItem
        for item in queue_items:
            itemFuture: ItemFuture = item.item_future
            input_data_path: Optional[str] = None
            try:
                input_data: Union[str, Image.Image] = itemFuture[item.input_names[0]]
                if isinstance(input_data, str):
                    input_data_path = input_data
                
                norm_config_to_use: Union[int, Dict[str, List[float]]] = self.normalization_config
                
                preprocessed_frame: Any = preprocess_image(input_data, self.image_size, self.use_half_precision, self.device, norm_config=norm_config_to_use)
                
                output_target = item.output_names[0] if isinstance(item.output_names, list) else item.output_names
                await itemFuture.set_data(output_target, preprocessed_frame)

            except FileNotFoundError as fnf_error:
                error_msg_path = input_data_path if input_data_path else "unknown_file (input was not a path)"
                self.logger.error(f"File not found error: {fnf_error} for file: {error_msg_path}")
                itemFuture.set_exception(fnf_error)
            except IOError as io_error:
                error_msg_path = input_data_path if input_data_path else "unknown_file (input was not a path)"
                self.logger.error(f"IO error (image might be corrupted): {io_error} for file: {error_msg_path}")
                itemFuture.set_exception(io_error)
            except Exception as e:
                error_msg_path = input_data_path if input_data_path else "unknown_file (input was not a path)"
                self.logger.error(f"An unexpected error occurred: {e} for file: {error_msg_path}", exc_info=True)
                itemFuture.set_exception(e)