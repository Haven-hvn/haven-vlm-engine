import logging
import torch
from lib.model.model import Model
from lib.model.ai_model_python.python_model import PythonModel as AiPythonModel
import time
from typing import Dict, Any, List, Optional, Union, Tuple, TextIO
from lib.async_lib.queue_item import QueueItem, ItemFuture

# Placeholder for ModelRunner if ai_processing is not available/inspectable
# If ModelRunner is a known class, replace Any with it.
ModelRunner = Any 

class AIModel(Model):
    def __init__(self, configValues: Dict[str, Any]):
        Model.__init__(self, configValues)
        self.max_model_batch_size: int = int(configValues.get("max_model_batch_size", 12))
        self.batch_size_per_VRAM_GB: Optional[float] = configValues.get("batch_size_per_VRAM_GB")
        self.model_file_name: Optional[str] = configValues.get("model_file_name")
        self.model_license_name: Optional[str] = configValues.get("model_license_name")
        self.model_threshold: Optional[float] = configValues.get("model_threshold")
        self.model_return_tags: bool = bool(configValues.get("model_return_tags", False))
        self.model_return_confidence: bool = bool(configValues.get("model_return_confidence", False))
        self.device: Optional[str] = configValues.get("device")
        self.fill_to_batch: bool = bool(configValues.get("fill_to_batch_size", True))
        self.model_image_size: Optional[Union[int, Tuple[int, int]]] = configValues.get("model_image_size")
        self.model_category: Optional[Union[str, List[str]]] = configValues.get("model_category")
        self.model_version: Optional[str] = str(configValues.get("model_version")) if configValues.get("model_version") is not None else None
        self.model_identifier: Optional[str] = configValues.get("model_identifier")
        # category_mappings maps internal model output index to pipeline's output list index for categories
        self.category_mappings: Optional[Dict[int, int]] = configValues.get("category_mappings")
        self.normalization_config: Union[int, Dict[str, List[float]]] = configValues.get("normalization_config", 1)
        
        if self.model_file_name is None:
            raise ValueError("model_file_name is required for models of type model")
        if self.model_category is not None and isinstance(self.model_category, list) and len(self.model_category) > 1:
            if self.category_mappings is None:
                raise ValueError("category_mappings is required for models with more than one category")
        
        self.model: Optional[Union[AiPythonModel, ModelRunner]] = None # AiPythonModel is for .pt, ModelRunner for .pt.enc
        self.tags: Dict[int, str] = {} # Initialized, to be loaded in self.load()

        if self.device is None:
            self.localdevice: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.localdevice: torch.device = torch.device(self.device)

        self.update_batch_with_mutli_models(1) # Initial call for single model case
    
    def update_batch_with_mutli_models(self, model_count: int) -> None:
        batch_multipliers: List[float] = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3] # Default up to 7 models
        # Ensure model_count is within the bounds of batch_multipliers
        effective_model_count: int = min(model_count, len(batch_multipliers))
        if effective_model_count == 0: effective_model_count = 1 # Should not happen, but safeguard

        if self.batch_size_per_VRAM_GB is not None and torch.cuda.is_available():
            try:
                multiplier: float = batch_multipliers[effective_model_count - 1]
                batch_size_temp: float = self.batch_size_per_VRAM_GB * multiplier
                gpuMemory: float = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                scaledBatchSize: int = custom_round(batch_size_temp * gpuMemory)
                # Ensure batch size is at least 1
                self.max_model_batch_size = max(1, scaledBatchSize) 
                self.max_batch_size = max(1, scaledBatchSize)
                self.max_queue_size = max(1, scaledBatchSize) # Or some other logic for queue size
                self.logger.debug(f"Setting batch size to {self.max_model_batch_size} based on VRAM size of {gpuMemory:.2f} GB for model {self.model_file_name} ({model_count} models active)")
            except Exception as e_vram:
                self.logger.error(f"Could not set batch size based on VRAM for {self.model_file_name}: {e_vram}. Using default {self.max_model_batch_size}.")
        # else: use pre-configured or default batch sizes

    async def worker_function(self, data: List[QueueItem]) -> None:
        if not data: return # No data to process
        try:
            # Assuming all images in a batch have the same shape
            # input_names[0] is assumed to be the image tensor
            first_image_tensor: torch.Tensor = data[0].item_future[data[0].input_names[0]]
            first_image_shape: torch.Size = first_image_tensor.shape
            
            # Create an empty tensor with the same shape as the input images
            # Ensure it matches the dtype of the input tensor as well
            images: torch.Tensor = torch.empty((len(data), *first_image_shape), dtype=first_image_tensor.dtype, device=self.localdevice)
            
            item: QueueItem
            for i, item in enumerate(data):
                itemFuture: ItemFuture = item.item_future
                images[i] = itemFuture[item.input_names[0]] # image tensor

            curr_time: float = time.time()
            # self.model must be loaded by this point
            if self.model is None or not hasattr(self.model, 'process_images'):
                 raise RuntimeError(f"Model {self.model_file_name} is not loaded or does not have process_images method.")
            
            model_results: Any = self.model.process_images(images) # type of model_results depends on model output
            self.logger.debug(f"Processed {len(images)} images in {time.time() - curr_time:.3f}s in {self.model_file_name}")

            for i, item in enumerate(data):
                item_future: ItemFuture = item.item_future
                # input_names[1] is threshold, input_names[2] is return_confidence
                threshold: Optional[float] = item_future.get(item.input_names[1], self.model_threshold)
                return_confidence_override: Optional[bool] = item_future.get(item.input_names[2])
                current_return_confidence: bool = return_confidence_override if return_confidence_override is not None else self.model_return_confidence
                
                # item.output_names is Union[str, List[str]]
                output_names_list: List[str] = [item.output_names] if isinstance(item.output_names, str) else item.output_names
                toReturn: Dict[str, List[Union[Tuple[str, float], str]]] = {output_name: [] for output_name in output_names_list}
                
                single_item_result: Any = model_results[i] # Result for the i-th image
                
                # Ensure tags are loaded
                if not self.tags:
                    self.logger.warning(f"Tags not loaded for model {self.model_file_name}. Skipping tag processing.")
                    continue

                confidence_value: torch.Tensor # Assuming result elements are tensors
                for j, confidence_value in enumerate(single_item_result):
                    tag_name: str = self.tags.get(j, f"unknown_tag_index_{j}")
                    processed_tag: Union[Tuple[str, float], str]

                    # Ensure threshold is float if not None
                    current_threshold: Optional[float] = float(threshold) if threshold is not None else None

                    if current_threshold is not None and confidence_value.item() > current_threshold:
                        if current_return_confidence:
                            processed_tag = (tag_name, round(confidence_value.item(), 2))
                        else:
                            processed_tag = tag_name
                    elif current_threshold is None: # No threshold, always include (if applicable based on model_return_tags)
                        if self.model_return_tags: # Check if tags should be returned at all without threshold
                            if current_return_confidence:
                                processed_tag = (tag_name, round(confidence_value.item(), 2)) # still provide confidence
                            else:
                                processed_tag = tag_name
                        else: continue # Skip if not returning tags and no threshold met
                    else: # Threshold exists but not met
                        continue
                    
                    # Map to category output lists
                    if self.category_mappings and j in self.category_mappings:
                        list_id_index: int = self.category_mappings[j]
                        if 0 <= list_id_index < len(output_names_list):
                            toReturn[output_names_list[list_id_index]].append(processed_tag)
                        else:
                            self.logger.warning(f"Category mapping index {list_id_index} out of bounds for output names.")
                    elif not self.category_mappings and output_names_list: # Default to first output list if no mapping
                        toReturn[output_names_list[0]].append(processed_tag)
                
                output_name_key: str
                result_val_list: List[Union[Tuple[str, float], str]]
                for output_name_key, result_val_list_val in toReturn.items():
                    await item_future.set_data(output_name_key, result_val_list_val)
        except Exception as e:
            self.logger.error(f"Error in AI model ({self.model_file_name}) worker_function: {e}", exc_info=True)
            # Propagate error to all item futures in the batch
            err_item: QueueItem
            for err_item in data:
                if hasattr(err_item, 'item_future') and err_item.item_future:
                    err_item.item_future.set_exception(e)

    async def load(self) -> None:
        if self.model is None:
            self.logger.info(f"Loading model {self.model_file_name} with batch size {self.max_model_batch_size}, queue {self.max_queue_size}, batch {self.max_batch_size}")
            model_path_base: str = f"./models/{self.model_file_name}"
            tags_path: str = f"{model_path_base}.tags.txt"
            
            try:
                if self.model_license_name is None:
                    # .pt model using AiPythonModel (custom class)
                    self.model = AiPythonModel(f"{model_path_base}.pt", self.max_model_batch_size, self.device, self.fill_to_batch)
                else:
                    # .pt.enc model using ModelRunner (external)
                    # Ensure ai_processing can be imported or handle gracefully
                    try:
                        from ai_processing import ModelRunner as ExternalModelRunner
                        self.model = ExternalModelRunner(f"{model_path_base}.pt.enc", f"./models/{self.model_license_name}.lic", self.max_model_batch_size, self.device)
                    except ImportError:
                        self.logger.error("Module 'ai_processing' for ModelRunner not found. Encrypted models cannot be loaded.")
                        raise RuntimeError("ai_processing module not found for encrypted model.")
                
                self.tags = get_index_to_tag_mapping(tags_path)
                # Default category mapping if single category and no explicit mapping
                if self.model_category and isinstance(self.model_category, str) and not self.category_mappings:
                    self.category_mappings = {i: 0 for i, _ in enumerate(self.tags)}
                elif self.model_category and isinstance(self.model_category, list) and len(self.model_category) == 1 and not self.category_mappings:
                     self.category_mappings = {i: 0 for i, _ in enumerate(self.tags)}

            except FileNotFoundError as fnf_e:
                self.logger.error(f"Failed to load model {self.model_file_name} or its tags: {fnf_e}")
                raise # Re-raise to signal load failure
            except Exception as load_e:
                self.logger.error(f"Exception loading model {self.model_file_name}: {load_e}", exc_info=True)
                raise # Re-raise
        elif hasattr(self.model, 'load_model') and callable(self.model.load_model): # For models that have explicit reload
            await self.model.load_model() # Assuming load_model might be async
        self.logger.info(f"Model {self.model_file_name} load process complete.")

def get_index_to_tag_mapping(path: str) -> Dict[int, str]:
    tags_txt_path: str = path
    index_to_tag: Dict[int, str] = {}
    try:
        with open(tags_txt_path, 'r', encoding='utf-8') as file_handle: # Renamed file to file_handle
            index: int
            line: str
            for index, line in enumerate(file_handle):
                index_to_tag[index] = line.strip()
    except FileNotFoundError:
        logging.getLogger("logger").error(f"Tags file not found at {tags_txt_path}. Tags will be empty.")
        # Return empty dict or raise, depending on desired behavior. Current behavior: returns empty.
    return index_to_tag

def custom_round(value: float) -> int:
    if value < 8:
        return int(value)
    remainder: int = int(value) % 8
    if remainder <= 5:
        return int(value) - remainder
    else:
        return int(value) + (8 - remainder)