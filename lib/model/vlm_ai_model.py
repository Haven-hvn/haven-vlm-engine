import logging
import torch
import time
from PIL import Image
import numpy as np
from lib.model.model import Model
from lib.model.ai_model import AIModel
from lib.model.vlm_model import VLMModel

class VLMAIModel(AIModel):
    def __init__(self, configValues):
        if "model_file_name" not in configValues:
            configValues["model_file_name"] = configValues.get("vlm_model_name", "HuggingFaceTB/SmolVLM-Instruct")
        super().__init__(configValues)
        self.max_model_batch_size = configValues.get("max_model_batch_size", 1)
        self.model_threshold = configValues.get("model_threshold", 0.5)
        self.model_return_tags = configValues.get("model_return_tags", True)
        self.model_return_confidence = configValues.get("model_return_confidence", True)
        self.device = configValues.get("device", None)
        self.model_category = configValues.get("model_category", None)
        self.model_version = configValues.get("model_version", "1.0")
        self.model_identifier = configValues.get("model_identifier", None)
        self.tag_list_path = configValues.get("tag_list_path", None)
        self.vlm_model_name = configValues.get("vlm_model_name", "HuggingFaceTB/SmolVLM-Instruct")
        self.use_quantization = configValues.get("use_quantization", True)
        
        if self.model_category is None:
            raise ValueError("model_category is required for VLM AI models")
        if self.tag_list_path is None:
            raise ValueError("tag_list_path is required for VLM AI models")
        
        self.logger = logging.getLogger("logger")
        self.vlm_model = None
        self.tags = []
        
        if self.device is None:
            self.localdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.localdevice = torch.device(self.device)

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
            self.logger.info(f"Loading VLM model {self.vlm_model_name}")
            
            # Load tag list
            self.tags = self._load_tag_list(self.tag_list_path)
            self.logger.info(f"Loaded {len(self.tags)} tags from {self.tag_list_path}")
            
            # Initialize VLM model
            self.vlm_model = VLMModel(
                tag_list=self.tags,
                model_name=self.vlm_model_name,
                use_quantization=self.use_quantization,
                device=self.device
            )
            self.logger.info(f"VLM model loaded successfully")
    
    def _load_tag_list(self, tag_list_path):
        """Load NSFW tags from a file"""
        tags = []
        try:
            with open(tag_list_path, 'r', encoding='utf-8') as f:
                tags = [line.strip() for line in f if line.strip()]
        except Exception as e:
            self.logger.error(f"Error loading tag list from {tag_list_path}: {e}")
            self.logger.debug("Stack trace:", exc_info=True)
            raise
        return tags
