from lib.async_lib.async_processing import QueueItem, ItemFuture
from lib.model.ai_model import AIModel
from lib.model.video_preprocessor import VideoPreprocessorModel
from lib.model.model_manager import ModelManager
from lib.pipeline.dynamic_ai_manager import DynamicAIManager
from lib.pipeline.model_wrapper import ModelWrapper
from typing import List, Dict, Any, Optional, Set, Union, Tuple
from lib.model.vlm_ai_model import VLMAIModel
import logging

logger: logging.Logger = logging.getLogger("logger")

class Pipeline:
    def __init__(self, configValues: Dict[str, Any], model_manager: ModelManager, dynamic_ai_manager: DynamicAIManager):
        if not validate_string_list(configValues.get("inputs")):
            raise ValueError("Error: Pipeline inputs must be a non-empty list of strings!")
        if not configValues.get("output") or not isinstance(configValues.get("output"), str):
            raise ValueError("Error: Pipeline output must be a non-empty string!")
        if not isinstance(configValues.get("models"), list) or not configValues.get("models"):
            raise ValueError("Error: Pipeline models must be a non-empty list!")
        
        self.short_name: Optional[str] = configValues.get("short_name")
        if self.short_name is None or not isinstance(self.short_name, str) or not self.short_name:
            raise ValueError("Error: Pipeline short_name must be a non-empty string!")
        self.version: Optional[Union[float, str]] = configValues.get("version")
        if self.version is None or not (isinstance(self.version, float) or isinstance(self.version, int) or (isinstance(self.version, str) and self.version)):
            raise ValueError("Error: Pipeline version must be a non-empty float or string representation of a number!")
        
        self.inputs: List[str] = configValues["inputs"]
        self.output: str = configValues["output"]

        self.models: List[ModelWrapper] = []
        model_config: Dict[str, Any]
        for model_config in configValues["models"]:
            if not validate_string_list(model_config.get("inputs")):
                raise ValueError("Error: Model inputs must be a non-empty list of strings!")
            if not model_config.get("name") or not isinstance(model_config.get("name"), str):
                raise ValueError("Error: Model name must be a non-empty string!")
            
            modelName: str = model_config["name"]
            model_inputs: List[str] = model_config["inputs"]
            model_outputs: Union[str, List[str]] = model_config.get("outputs")
            if model_outputs is None or (isinstance(model_outputs, list) and not model_outputs and not all(isinstance(o, str) for o in model_outputs)) or (isinstance(model_outputs, str) and not model_outputs) :
                if modelName != "result_finisher":
                    pass

            if modelName == "dynamic_video_ai":
                dynamic_models: List[ModelWrapper] = dynamic_ai_manager.get_dynamic_video_ai_models(model_inputs, model_outputs if isinstance(model_outputs, list) else [model_outputs] if model_outputs else [])
                self.models.extend(dynamic_models)
                continue
            elif modelName == "dynamic_image_ai":
                dynamic_models: List[ModelWrapper] = dynamic_ai_manager.get_dynamic_image_ai_models(model_inputs, model_outputs if isinstance(model_outputs, list) else [model_outputs] if model_outputs else [])
                self.models.extend(dynamic_models)
                continue
            
            returned_model: Any = model_manager.get_or_create_model(modelName)
            self.models.append(ModelWrapper(returned_model, model_inputs, model_outputs, model_name_for_logging=modelName))

        categories_set: Set[str] = set()
        wrapper_model: ModelWrapper
        for wrapper_model in self.models:
            if hasattr(wrapper_model.model, 'model') and isinstance(wrapper_model.model.model, AIModel):
                current_categories: Union[str, List[str]] = wrapper_model.model.model.model_category
                if isinstance(current_categories, str):
                    if current_categories in categories_set:
                        raise ValueError(f"Error: AI models must not have overlapping categories! Category: {current_categories}")
                    categories_set.add(current_categories)
                elif isinstance(current_categories, list):
                    cat: str
                    for cat in current_categories:
                        if cat in categories_set:
                            raise ValueError(f"Error: AI models must not have overlapping categories! Category: {cat}")
                        categories_set.add(cat)
        
        # Determine if this is a VLM pipeline and configure VideoPreprocessorModels accordingly
        is_vlm_pipeline: bool = any(isinstance(mw.model.model, VLMAIModel) for mw in self.models if hasattr(mw.model, 'model'))
        if is_vlm_pipeline:
            for model_wrapper in self.models:
                if hasattr(model_wrapper.model, 'model') and isinstance(model_wrapper.model.model, VideoPreprocessorModel):
                    model_wrapper.model.model.set_vlm_pipeline_mode(True)
    
    async def event_handler(self, itemFuture: ItemFuture, key: str) -> None:
        if key == self.output:
            if key in itemFuture:
                itemFuture.close_future(itemFuture[key])
            else:
                pass
        
        current_model_wrapper: ModelWrapper
        for current_model_wrapper in self.models:
            if key in current_model_wrapper.inputs:
                allOtherInputsPresent: bool = True

                inputName: str
                for inputName in current_model_wrapper.inputs:
                    if inputName != key:
                        is_present = (itemFuture.data is not None and inputName in itemFuture.data)
                        if not is_present:
                            allOtherInputsPresent = False
                            break
                
                if allOtherInputsPresent:
                    await current_model_wrapper.model.add_to_queue(QueueItem(itemFuture, current_model_wrapper.inputs, current_model_wrapper.outputs))

    async def start_model_processing(self) -> None:
        model_wrapper: ModelWrapper
        for model_wrapper in self.models:
            if hasattr(model_wrapper.model, 'start_workers') and callable(model_wrapper.model.start_workers):
                 await model_wrapper.model.start_workers()

    def get_first_video_preprocessor(self) -> Optional[VideoPreprocessorModel]:
        model_wrapper: ModelWrapper
        for model_wrapper in self.models:
            if hasattr(model_wrapper.model, 'model') and isinstance(model_wrapper.model.model, VideoPreprocessorModel):
                return model_wrapper.model.model
        return None
    
    def get_first_ai_model(self) -> Optional[AIModel]:
        model_wrapper: ModelWrapper
        for model_wrapper in self.models:
            if hasattr(model_wrapper.model, 'model') and isinstance(model_wrapper.model.model, AIModel):
                return model_wrapper.model.model
        return None
    
    def get_ai_models_info(self) -> List[Tuple[Union[str, float, None], Optional[str], Optional[str], Optional[Union[str, List[str]]]]]:
        ai_version_and_ids: List[Tuple[Union[str, float, None], Optional[str], Optional[str], Optional[Union[str, List[str]]]]] = []
        model_wrapper: ModelWrapper
        for model_wrapper in self.models:
            if hasattr(model_wrapper.model, 'model') and isinstance(model_wrapper.model.model, AIModel):
                inner_ai_model: AIModel = model_wrapper.model.model
                version = getattr(inner_ai_model, 'model_version', None)
                identifier = getattr(inner_ai_model, 'model_identifier', None)
                file_name = getattr(inner_ai_model, 'model_file_name', None)
                category = getattr(inner_ai_model, 'model_category', None)
                ai_version_and_ids.append((version, identifier, file_name, category))
        return ai_version_and_ids

def validate_string_list(input_list: Any) -> bool:
    if not isinstance(input_list, list):
        return False
    item: Any
    for item in input_list:
        if not isinstance(item, str):
            return False
    return True
    