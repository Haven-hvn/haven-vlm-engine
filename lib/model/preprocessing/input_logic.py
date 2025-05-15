import logging
from typing import Tuple, Set, List, Dict, Any, Optional, Union
from lib.model.postprocessing.post_processing_settings import post_processing_config
from lib.model.postprocessing.AI_VideoResult import AIVideoResult
from lib.pipeline.pipeline import Pipeline

logger: logging.Logger = logging.getLogger("logger")

def process_video_preprocess(video_result: AIVideoResult, frame_interval: Optional[float], threshold: Optional[float], pipeline: Pipeline) -> Tuple[bool, Set[str]]:
    ai_workNeeded: bool = False
    ai_models_info: List[Tuple[Optional[Union[str, float]], Optional[str], Optional[str], Optional[Union[str, List[str]]]]] = pipeline.get_ai_models_info()

    skipped_model_categories: Set[str] = set()
    previouslyUsedModelsDict: Dict[str, Any] = video_result.metadata.models
    
    ai_version: Optional[Union[str, float]]
    ai_id: Optional[str]
    ai_filename: Optional[str]
    ai_categories: Optional[Union[str, List[str]]]
    category: str
    
    for ai_version, ai_id, ai_filename, ai_categories_raw in ai_models_info:
        current_ai_categories: List[str] = []
        if isinstance(ai_categories_raw, str):
            current_ai_categories = [ai_categories_raw]
        elif isinstance(ai_categories_raw, list):
            current_ai_categories = [str(cat) for cat in ai_categories_raw]

        for category in current_ai_categories:
            if category not in previouslyUsedModelsDict:
                ai_workNeeded = True
            else:
                previousUsedModel: Any = previouslyUsedModelsDict[category]

                current_frame_interval: float = frame_interval if frame_interval is not None else 0.0
                current_threshold: float = threshold if threshold is not None else 0.0
                current_ai_id_int: int
                try:
                    current_ai_id_int = int(str(ai_id)) if ai_id is not None and str(ai_id).isdigit() else 0
                except ValueError:
                    current_ai_id_int = 0
                
                current_ai_version_float: float
                try:
                    current_ai_version_float = float(str(ai_version)) if ai_version is not None else 0.0
                except ValueError:
                    current_ai_version_float = 0.0

                needs_reprocessed: int = previousUsedModel.needs_reprocessed(current_frame_interval, current_threshold, current_ai_version_float, current_ai_id_int, ai_filename)
                if post_processing_config.get('reprocess_with_more_accurate_same_model', False) and needs_reprocessed == 1:
                    ai_workNeeded = True
                elif needs_reprocessed == 2:
                    ai_workNeeded = True
                else:
                    skipped_model_categories.add(category)
    return ai_workNeeded, skipped_model_categories