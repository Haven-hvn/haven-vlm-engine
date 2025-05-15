import logging
from lib.async_lib.async_processing import ModelProcessor
from lib.config.config_utils import load_config
from lib.model.ai_model import AIModel
from lib.model.python_model import PythonModel
from lib.model.video_preprocessor import VideoPreprocessorModel
from lib.model.image_preprocessor import ImagePreprocessorModel
from lib.model.vlm_ai_model import VLMAIModel
from typing import Dict, Any, List, Optional


class ModelManager:
    def __init__(self):
        self.models: Dict[str, ModelProcessor] = {}
        self.logger: logging.Logger = logging.getLogger("logger")
        self.ai_models: List[ModelProcessor] = []

    def get_or_create_model(self, modelName: str) -> ModelProcessor:
        if modelName not in self.models:
            created_model: Optional[ModelProcessor] = self.create_model(modelName)
            if created_model is None:
                raise ValueError(f"Failed to create model: {modelName}")
            self.models[modelName] = created_model
        return self.models[modelName]
    
    def get_and_refresh_model(self, modelName: str) -> ModelProcessor:
        created_model: Optional[ModelProcessor] = self.create_model(modelName)
        if created_model is None:
            raise ValueError(f"Failed to create refreshed model: {modelName}")
        if modelName in self.models:
            pass 
        self.models[modelName] = created_model
        return self.models[modelName]
    
    def create_model(self, modelName: str) -> Optional[ModelProcessor]:
        if not isinstance(modelName, str):
            self.logger.error("Model name must be a string.")
            raise ValueError("Model names must be strings that are the name of the model config file!")
        
        model_config_path: str = f"./config/models/{modelName}.yaml"
        try:
            loaded_model_config: Dict[str, Any] = load_config(model_config_path)
            model_processor_instance: ModelProcessor = self.model_factory(loaded_model_config)
            return model_processor_instance
        except FileNotFoundError:
            self.logger.error(f"Model configuration file not found: {model_config_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading or creating model from {model_config_path}: {e}")
            self.logger.debug("Stack trace:", exc_info=True)
            return None
    
    def model_factory(self, model_config: Dict[str, Any]) -> ModelProcessor:
        model_type: str = model_config.get("type", "")
        
        model_instance: Any
        match model_type:
            case "video_preprocessor":
                model_instance = VideoPreprocessorModel(model_config)
                return ModelProcessor(model_instance)
            case "image_preprocessor":
                model_instance = ImagePreprocessorModel(model_config)
                return ModelProcessor(model_instance)
            case "model":
                model_instance = AIModel(model_config)
                model_processor: ModelProcessor = ModelProcessor(model_instance)
                self.ai_models.append(model_processor)
                model_count: int = len(self.ai_models)
                if model_count > 1:
                    proc: ModelProcessor
                    for proc in self.ai_models:
                        if isinstance(proc.model, AIModel) and hasattr(proc.model, 'update_batch_with_mutli_models'):
                            proc.model.update_batch_with_mutli_models(model_count)
                            proc.update_values_from_child_model()
                return model_processor
            case "vlm_model":
                model_instance = VLMAIModel(model_config)
                model_processor: ModelProcessor = ModelProcessor(model_instance)
                self.ai_models.append(model_processor)
                return model_processor
            case "python":
                model_instance = PythonModel(model_config)
                return ModelProcessor(model_instance)
            case _:
                raise ValueError(f"Model type '{model_type}' not recognized!")
