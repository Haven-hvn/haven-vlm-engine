import logging
from lib.config.config_utils import load_config
from lib.model.ai_model import AIModel
from lib.model.model_manager import ModelManager
from lib.pipeline.pipeline import ModelWrapper
from lib.migrations.migration_v20 import migrate_to_2_0
from lib.server.exceptions import NoActiveModelsException
from typing import List, Dict, Any, Optional, Union, Tuple


ai_active_directory: str = "./config/active_ai.yaml"



class DynamicAIManager:
    def __init__(self, model_manager: ModelManager):
        self.model_manager: ModelManager = model_manager
        self.configfile: Dict[str, Any] = load_config(ai_active_directory)
        self.ai_model_names: List[str] = self.configfile.get("active_ai_models", [])
        self.loaded: bool = False
        self.image_size: Optional[Union[Tuple[int, int], List[int]]] = None
        self.normalization_config: Optional[Dict[str, Any]] = None
        self.models: List[AIModel] = []
        self.logger: logging.Logger = logging.getLogger("logger")

    def load(self) -> None:
        if self.loaded:
            return
        models: List[AIModel] = []

        if self.ai_model_names is None or len(self.ai_model_names) == 0:
            self.logger.error("Error: No active AI models found in active_ai.yaml")
            raise NoActiveModelsException("Error: No active AI models found in active_ai.yaml")
        for model_name in self.ai_model_names:
            models.append(self.model_manager.get_or_create_model(model_name))
        self.__verify_models(models)
        self.models = models
        self.loaded = True

    def get_dynamic_video_ai_models(self, inputs: List[str], outputs: List[str]) -> List[ModelWrapper]:
        self.load()
        model_wrappers: List[ModelWrapper] = []
        video_preprocessor: AIModel = self.model_manager.get_or_create_model("video_preprocessor_dynamic")
        video_preprocessor.model.image_size = self.image_size
        video_preprocessor.model.normalization_config = self.normalization_config

        # add a preprocessor
        model_wrappers.append(ModelWrapper(video_preprocessor, inputs, ["dynamic_children", "dynamic_frame", "frame_index", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"]))

        # add the ai models
        model: AIModel
        for model in self.models:
            model_wrappers.append(ModelWrapper(model, ["dynamic_frame", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"], model.model.model_category))

        # coalesce all the ai results
        coalesce_inputs: List[str] = []
        for model in self.models:
            categories: Union[str, List[str]] = model.model.model_category
            if isinstance(categories, list):
                coalesce_inputs.extend(categories)
            else:
                coalesce_inputs.append(categories)
        coalesce_inputs.insert(0, "frame_index")
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_coalescer"), coalesce_inputs, ["dynamic_result"]))

        # finish results
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_finisher"), ["dynamic_result"], []))

        # await children
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("batch_awaiter"), ["dynamic_children"], outputs))
        self.logger.debug("Finished creating dynamic Video AI models")
        return model_wrappers
    
    def get_dynamic_image_ai_models(self, inputs: List[str], outputs: List[str]) -> List[ModelWrapper]:
        self.load()
        model_wrappers: List[ModelWrapper] = []
        image_preprocessor: AIModel = self.model_manager.get_or_create_model("image_preprocessor_dynamic")
        image_preprocessor.model.image_size = self.image_size
        image_preprocessor.model.normalization_config = self.normalization_config

        # add a preprocessor
        model_wrappers.append(ModelWrapper(image_preprocessor, [inputs[0]], ["dynamic_image"]))

        # add the ai models
        model: AIModel
        for model in self.models:
            model_wrappers.append(ModelWrapper(model, ["dynamic_image", inputs[1], inputs[2], inputs[3]], model.model.model_category))

        # coalesce all the ai results
        coalesce_inputs: List[str] = []
        for model in self.models:
            categories: Union[str, List[str]] = model.model.model_category
            if isinstance(categories, list):
                coalesce_inputs.extend(categories)
            else:
                coalesce_inputs.append(categories)
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_coalescer"), coalesce_inputs, outputs=outputs))

        self.logger.debug("Finished creating dynamic Image AI models")
        return model_wrappers
    
    def __verify_models(self, models: List[AIModel], second_pass: bool = False) -> None:
        current_image_size: Optional[Union[Tuple[int, int], List[int]]] = None
        current_norm_config: Optional[Dict[str, Any]] = None
        
        reverify_models_list: List[AIModel] = []

        model_instance: AIModel
        for model_instance in models:
            inner_model: AIModel = model_instance.model
            if not isinstance(inner_model, AIModel):
                raise ValueError(f"Error: Dynamic AI models must all be AI models! {inner_model} is not an AI model!")
            try:
                if inner_model.model_category is None:
                    raise ValueError(f"Error: Dynamic AI models must have model_category set! {inner_model} does not have model_category set and needs a migration")
                if inner_model.model_version is None:
                    raise ValueError(f"Error: Dynamic AI models must have model_version set! {inner_model} does not have model_version set and needs a migration")
                if inner_model.model_image_size is None:
                    raise ValueError(f"Error: Dynamic AI models must have model_image_size set! {inner_model} does not have model_image_size set and needs a migration")
            except ValueError as e:
                if second_pass:
                    raise e
                self.logger.warning(f"Detected old model files. Attempting migration...")
                migrate_to_2_0()
                self.logger.info("Migration complete. Verifying models again...")
                reverify_models_list.clear()
                model_name: str
                for model_name in self.ai_model_names:
                    reverify_models_list.append(self.model_manager.get_and_refresh_model(model_name))
                self.__verify_models(reverify_models_list, True)
                return
            
            if current_image_size is None:
                current_image_size = inner_model.model_image_size
            elif current_image_size != inner_model.model_image_size:
                raise ValueError(f"Error: Dynamic AI models must all have the same model_image_size! {inner_model} has a different model_image_size than other models!")
            
            if current_norm_config is None:
                current_norm_config = inner_model.normalization_config
            elif current_norm_config != inner_model.normalization_config:
                raise ValueError(f"Error: Dynamic AI models must all have the same normalization_config! {inner_model} has a different normalization_config than other models!")
        self.image_size = current_image_size
        self.normalization_config = current_norm_config
        self.logger.debug("Finished verifying dynamic AI models")

        

