import logging
# import torch # Potentially remove if localdevice and tensor conversion are no longer needed
import time
from PIL import Image
import numpy as np # Keep for now for tensor to PIL conversion
from lib.model.model import Model
from lib.model.ai_model import AIModel
# Updated import to the new client
from lib.model.vlm_model import OpenAICompatibleVLMClient
from typing import Dict, Any, List, Optional, Union, Tuple # Added imports
from lib.async_lib.async_processing import ItemFuture, QueueItem # Added QueueItem

class VLMAIModel(AIModel):
    def __init__(self, configValues: Dict[str, Any]):
        # Ensure base AIModel init is called, model_file_name might not be relevant for remote API
        # but AIModel base class might expect it.
        # If "model_file_name" is not used by OpenAICompatibleVLMClient, 
        # it could be set to a placeholder or derived from model_id if necessary for AIModel.
        if "model_file_name" not in configValues and "model_id" in configValues:
            configValues["model_file_name"] = configValues["model_id"] # Or a more suitable placeholder
        elif "model_file_name" not in configValues:
            configValues["model_file_name"] = "remote_vlm_client" # Default placeholder
            
        super().__init__(configValues)
        
        # Store the full config for the client, or select specific keys
        # For now, assume configValues directly contains all necessary client settings
        self.client_config: Dict[str, Any] = configValues 

        self.max_model_batch_size: int = int(configValues.get("max_model_batch_size", 1))
        self.model_threshold: float = float(configValues.get("model_threshold", 0.5))
        self.model_return_confidence: bool = bool(configValues.get("model_return_confidence", True))
        self.model_category: Union[str, List[str]] = configValues.get("model_category")
        self.model_version: str = str(configValues.get("model_version", "1.0"))
        self.model_identifier: str = str(configValues.get("model_identifier", "default_vlm_identifier"))
        
        # Fields like tag_list_path, vlm_model_name, use_quantization, device are now part of client_config
        # and handled by OpenAICompatibleVLMClient itself.

        if self.model_category is None:
            raise ValueError("model_category is required for VLM AI models")
        # tag_list_path check is now handled by OpenAICompatibleVLMClient

        self.logger: logging.Logger = logging.getLogger("logger")
        self.vlm_model: Optional[OpenAICompatibleVLMClient] = None
        
        # localdevice might not be needed if all VLM work is remote.
        # For now, kept for the tensor to PIL conversion logic.
        # If image_tensor can be received as PIL, this (and torch import) can be removed.
        # self.localdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # Consider removing self.device and self.localdevice if not used elsewhere.

    async def worker_function(self, data: List[QueueItem]):
        try:
            item: QueueItem # Type for loop variable
            for i, item in enumerate(data):
                itemFuture: ItemFuture = item.item_future
                image_tensor: Any = itemFuture[item.input_names[0]] # Assuming torch.Tensor like object
                
                # threshold: float = itemFuture.get(item.input_names[1], self.model_threshold) # Use get for safety
                threshold_val: Optional[float] = itemFuture[item.input_names[1]] if len(item.input_names) > 1 and item.input_names[1] is not None else None
                threshold: float = threshold_val if threshold_val is not None else self.model_threshold
                
                # Ensure threshold is float
                if not isinstance(threshold, float):
                    threshold = self.model_threshold

                return_confidence: bool = self.model_return_confidence
                # Check if item.input_names[2] exists and is not None
                # if len(item.input_names) > 2 and item.input_names[2] is not None and itemFuture.get(item.input_names[2]) is not None:
                #    return_confidence = itemFuture[item.input_names[2]]
                if len(item.input_names) > 2 and item.input_names[2] is not None:
                    confidence_val: Optional[bool] = itemFuture[item.input_names[2]]
                    if confidence_val is not None:
                        return_confidence = confidence_val
                
                image_np: np.ndarray
                if hasattr(image_tensor, 'cpu') and hasattr(image_tensor, 'numpy'): # Check if it's a tensor
                    image_np = image_tensor.cpu().numpy()
                elif isinstance(image_tensor, np.ndarray):
                    image_np = image_tensor # Already a numpy array
                elif isinstance(image_tensor, Image.Image): # If it's already a PIL Image
                    image_pil: Image.Image = image_tensor # Skip conversion
                    # Directly process if it's already a PIL image
                    curr_time_pil: float = time.time()
                    scores_pil: Dict[str, float] = self.vlm_model.analyze_frame(image_pil)
                    self.logger.debug(f"Processed PIL image with VLM in {time.time() - curr_time_pil}s")
                    # ... (rest of the logic for PIL image, similar to below)
                else:
                    self.logger.error(f"Unsupported image_tensor type: {type(image_tensor)}")
                    # Handle error or skip item
                    await itemFuture.set_exception(TypeError(f"Unsupported image_tensor type: {type(image_tensor)}"))
                    continue

                # Handle different tensor formats if not already a PIL image
                if not isinstance(image_tensor, Image.Image):
                    if image_np.ndim == 3 and image_np.shape[0] == 3:  # CHW
                        image_np = np.transpose(image_np, (1, 2, 0)) # HWC
                    elif image_np.ndim == 2: # Grayscale H W
                         # If grayscale, VLM might expect 3 channels. Stack it or handle as is.
                         # For now, let's assume it needs to be converted to RGB for PIL fromarray
                        image_np = np.stack((image_np,)*3, axis=-1)

                    # Ensure data is in range [0, 1] if it's float, then scale to [0, 255]
                    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                        if image_np.min() >= 0 and image_np.max() <= 1:
                            image_np = (image_np * 255)
                        # else: values are not in [0,1], could be already [0,255] or other range.
                        # logger might be useful here if assumptions about range are critical.
                    
                    image_np = image_np.astype(np.uint8)
                    image_pil: Image.Image = Image.fromarray(image_np)
                
                curr_time: float = time.time()
                scores: Dict[str, float] = self.vlm_model.analyze_frame(image_pil)
                self.logger.debug(f"Processed image with VLM in {time.time() - curr_time}s")
                
                # Ensure output_names is a list
                current_output_names: List[str]
                if isinstance(item.output_names, str):
                    current_output_names = [item.output_names]
                elif isinstance(item.output_names, list):
                    current_output_names = item.output_names
                else: # Should not happen based on ModelWrapper
                    self.logger.error(f"Unexpected type for item.output_names: {type(item.output_names)}")
                    await itemFuture.set_exception(TypeError("Invalid output_names type"))
                    continue

                toReturn: Dict[str, List[Union[Tuple[str, float], str]]] = {output_name: [] for output_name in current_output_names}
                
                tag_name: str
                confidence: float
                for tag_name, confidence in scores.items():
                    if confidence > threshold:
                        tag: Union[Tuple[str, float], str]
                        if return_confidence:
                            tag = (tag_name, round(confidence, 2))
                        else:
                            tag = tag_name
                        
                        # Add to the appropriate category based on self.model_category and current_output_names
                        # This logic assumes self.model_category maps to or is contained in current_output_names
                        # For simplicity, if model_category is a list, we take the first. This might need refinement.
                        target_output_name: str
                        if isinstance(self.model_category, list):
                            if self.model_category: # Ensure list is not empty
                                target_output_name = self.model_category[0]
                            else: # Should not happen if validation is done in __init__
                                self.logger.warning("Model category list is empty.")
                                continue # Skip this tag
                        else: # string
                            target_output_name = self.model_category

                        if target_output_name in toReturn:
                             toReturn[target_output_name].append(tag)
                        elif current_output_names: # Fallback to the first output name if specific category not found
                            toReturn[current_output_names[0]].append(tag)

                output_name_key: str
                result_list: List[Union[Tuple[str, float], str]]
                for output_name_key, result_list_val in toReturn.items():
                    await itemFuture.set_data(output_name_key, result_list_val)
            self.logger.info(f"Processed {len(data)} items with VLM AI model {self.model_identifier}")
            # the length of data is always 1
                    
                
        except Exception as e:
            self.logger.error(f"Error in VLM AI model worker_function: {e}")
            self.logger.debug(f"Error in {self.model_identifier}")
            self.logger.debug("Stack trace:", exc_info=True)
            item_err: QueueItem # type for loop var
            for item_err in data: # item was already defined
                item_err.item_future.set_exception(e)

    async def load(self) -> None:
        if self.vlm_model is None:
            self.logger.info(f"Loading VLM client for model_id: {self.client_config.get('model_id')}")
            
            # OpenAICompatibleVLMClient handles its own tag loading via its config.
            # No need to load tags separately here.
            
            try:
                # Pass the client_config dictionary to the client
                self.vlm_model = OpenAICompatibleVLMClient(config=self.client_config)
                self.logger.info(f"OpenAI VLM client for {self.client_config.get('model_id')} loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAICompatibleVLMClient: {e}", exc_info=True)
                raise # Re-raise the exception to signal failure to load
    
    # _load_tag_list method is no longer needed as the client handles it.
    # def _load_tag_list(self, tag_list_path):
    #     """Load NSFW tags from a file"""
    #     tags = []
    #     try:
    #         with open(tag_list_path, 'r', encoding='utf-8') as f:
    #             tags = [line.strip() for line in f if line.strip()]
    #     except Exception as e:
    #         self.logger.error(f"Error loading tag list from {tag_list_path}: {e}")
    #         self.logger.debug("Stack trace:", exc_info=True)
    #         raise
    #     return tags
