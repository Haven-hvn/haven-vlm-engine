import logging
from lib.async_lib.async_processing import ItemFuture
from lib.config.config_utils import load_config
from lib.pipeline.dynamic_ai_manager import DynamicAIManager
from lib.pipeline.pipeline import Pipeline
from lib.model.model_manager import ModelManager
from lib.server.exceptions import NoActiveModelsException, ServerStopException
from typing import Dict, List, Any

class PipelineManager:
    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.logger: logging.Logger = logging.getLogger("logger")
        self.model_manager: ModelManager = ModelManager()
        self.dynamic_ai_manager: DynamicAIManager = DynamicAIManager(self.model_manager)
    
    async def load_pipelines(self, pipeline_strings: List[str]):
        pipeline_str: str
        for pipeline_str in pipeline_strings:
            self.logger.info(f"Loading pipeline: {pipeline_str}")
            if not isinstance(pipeline_str, str):
                raise ValueError("Pipeline names must be strings that are the name of the pipeline config file!")
            pipeline_config_path: str = f"./config/pipelines/{pipeline_str}.yaml"
            try:
                loaded_config: Dict[str, Any] = load_config(pipeline_config_path)
                newpipeline: Pipeline = Pipeline(loaded_config, self.model_manager, self.dynamic_ai_manager)
                self.pipelines[pipeline_str] = newpipeline
                await newpipeline.start_model_processing()
                self.logger.info(f"Pipeline {pipeline_str} V{newpipeline.version} loaded successfully!")
            except NoActiveModelsException as e_no_models:
                raise e_no_models
            except Exception as e_general:
                if pipeline_str in self.pipelines:
                    del self.pipelines[pipeline_str]
                self.logger.error(f"Error loading pipeline {pipeline_str}: {e_general}")
                self.logger.debug("Exception details:", exc_info=True)
            
        if not self.pipelines:
            raise ServerStopException("Error: No valid pipelines loaded!")
            
    def get_pipeline(self, pipeline_name: str) -> Pipeline:
        if not pipeline_name in self.pipelines:
            self.logger.error(f"Error: Pipeline: {pipeline_name} not found in valid loaded pipelines!")
            raise ValueError(f"Error: Pipeline: {pipeline_name} not found in valid loaded pipelines!")
        pipeline: Pipeline = self.pipelines[pipeline_name]
        return pipeline

    async def get_request_future(self, data: List[Any], pipeline_name: str) -> ItemFuture:
        if not pipeline_name in self.pipelines:
            self.logger.error(f"Error: Pipeline: {pipeline_name} not found in valid loaded pipelines!")
            raise ValueError(f"Error: Pipeline: {pipeline_name} not found in valid loaded pipelines!")
        pipeline: Pipeline = self.pipelines[pipeline_name]
        futureData: Dict[str, Any] = {}
        if len(data) != len(pipeline.inputs):
            self.logger.error(f"Error: Data length does not match pipeline inputs length for pipeline {pipeline_name}!")
            raise ValueError(f"Error: Data length does not match pipeline inputs length for pipeline {pipeline_name}!")
        
        inputName: str
        inputData: Any
        for inputName, inputData in zip(pipeline.inputs, data):
            futureData[inputName] = inputData
        futureData["pipeline"] = pipeline
        itemFuture: ItemFuture = await ItemFuture.create(None, futureData, pipeline.event_handler)
        return itemFuture