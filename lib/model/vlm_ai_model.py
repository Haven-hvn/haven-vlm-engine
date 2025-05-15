import logging
# import torch # Potentially remove if localdevice and tensor conversion are no longer needed
import time
from PIL import Image
import numpy as np # Keep for now for tensor to PIL conversion
from lib.model.model import Model
from lib.model.ai_model import AIModel
# Updated import to the new client
from lib.model.vlm_model import OpenAICompatibleVLMClient

class VLMAIModel(AIModel):
    def __init__(self, configValues):
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
        self.client_config = configValues 

        self.max_model_batch_size = int(configValues.get("max_model_batch_size", 1))
        self.model_threshold = float(configValues.get("model_threshold", 0.5))
        self.model_return_tags = bool(configValues.get("model_return_tags", True))
        self.model_return_confidence = bool(configValues.get("model_return_confidence", True))
        self.model_category = configValues.get("model_category")
        self.model_version = str(configValues.get("model_version", "1.0"))
        self.model_identifier = str(configValues.get("model_identifier", "default_vlm_identifier"))
        
        # Fields like tag_list_path, vlm_model_name, use_quantization, device are now part of client_config
        # and handled by OpenAICompatibleVLMClient itself.

        if self.model_category is None:
            raise ValueError("model_category is required for VLM AI models")
        # tag_list_path check is now handled by OpenAICompatibleVLMClient

        self.logger = logging.getLogger("logger")
        self.vlm_model = None
        
        # localdevice might not be needed if all VLM work is remote.
        # For now, kept for the tensor to PIL conversion logic.
        # If image_tensor can be received as PIL, this (and torch import) can be removed.
        # self.localdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # Consider removing self.device and self.localdevice if not used elsewhere.

    async def worker_function(self, data):
        try:
            # Process each item in the batch
            for i, item in enumerate(data):
                itemFuture = item.item_future
                image_tensor = itemFuture[item.input_names[0]]
                threshold = itemFuture[item.input_names[1]] or self.model_threshold
                return_confidence = self.model_return_confidence
                if itemFuture[item.input_names[2]] is not None:
                    return_confidence = itemFuture[item.input_names[2]]
                
                # Convert tensor to PIL Image for VLM processing
                # Note: This conversion might need adjustment based on your preprocessing
                image_np = image_tensor.cpu().numpy()
                
                # Handle different tensor formats
                if image_np.shape[0] == 3:  # If in CHW format (channels, height, width)
                    image_np = np.transpose(image_np, (1, 2, 0))
                
                # Convert to uint8 for PIL
                image_np = (image_np * 255).astype('uint8')
                image_pil = Image.fromarray(image_np)
                
                # Process with VLM model
                curr = time.time()
                scores = self.vlm_model.analyze_frame(image_pil)
                self.logger.debug(f"Processed image with VLM in {time.time() - curr}s")
                
                # Format results for the existing pipeline
                toReturn = {output_name: [] for output_name in item.output_names}
                
                # Convert scores to the format expected by the existing pipeline
                for tag_name, confidence in scores.items():
                    if confidence > threshold:
                        if return_confidence:
                            tag = (tag_name, round(confidence, 2))
                        else:
                            tag = tag_name
                        
                        # Add to the appropriate category
                        if isinstance(self.model_category, list):
                            # If multiple categories, add to the first one
                            # This might need adjustment based on your needs
                            toReturn[item.output_names[0]].append(tag)
                        else:
                            toReturn[item.output_names[0]].append(tag)
                
                # Set the results
                for output_name, result_list in toReturn.items():
                    await itemFuture.set_data(output_name, result_list)
                
        except Exception as e:
            self.logger.error(f"Error in VLM AI model worker_function: {e}")
            self.logger.debug(f"Error in {self.model_identifier}")
            self.logger.debug("Stack trace:", exc_info=True)
            for item in data:
                item.item_future.set_exception(e)

    async def load(self):
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
