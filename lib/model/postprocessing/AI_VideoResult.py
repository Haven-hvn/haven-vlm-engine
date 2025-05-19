import gzip
import logging
from typing import Dict, List, Optional, Tuple, Set, Any, Union, TYPE_CHECKING
from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo
import math

if TYPE_CHECKING:
    from lib.model.postprocessing.AI_VideoResultV0 import AIVideoResultV0

logger: logging.Logger = logging.getLogger("logger")

class TagTimeFrame(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    start: int
    end: Optional[int] = None
    confidence: Optional[float] = None

    @field_validator('start', 'end', mode='before')
    @classmethod
    def _floor_and_convert_to_int(cls, v: Any, info: ValidationInfo) -> Optional[int]:
        if v is None and info.field_name == 'end':
            return None
        if v is None:
            raise ValueError(f"Field '{info.field_name}' received None but is not Optional or handled.")

        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(math.floor(v))
        try:
            return int(math.floor(float(str(v))))
        except (ValueError, TypeError) as e:
            logger.error(f"TagTimeFrame validator: Could not convert '{v}' for '{info.field_name}' to int: {e}")
            raise ValueError(f"Invalid value '{v}' for field '{info.field_name}', cannot convert to integer.") from e

    def __str__(self) -> str:
        return f"TagTimeFrame(start={self.start}, end={self.end}, confidence={self.confidence})"

class ModelInfo(BaseModel):
    frame_interval: float
    threshold: float
    version: float
    ai_model_id: int
    file_name: Optional[str] = None
    
    def needs_reprocessed(self, 
                          new_frame_interval: float, 
                          new_threshold: float, 
                          new_version: float,
                          new_ai_model_id: int, 
                          new_file_name: Optional[str]
                         ) -> int:
        model_toReturn: int = -1

        if new_file_name == self.file_name and new_version == self.version:
            model_toReturn = 0
        elif new_version == self.version and new_ai_model_id < self.ai_model_id and self.ai_model_id >= 950:
            model_toReturn = 2
        elif new_version == self.version and new_ai_model_id < self.ai_model_id:
            model_toReturn = 1
        elif new_version == self.version and new_ai_model_id >= self.ai_model_id:
            model_toReturn = 0
        else:
            model_toReturn = 2

        same_config: bool = True

        if self.frame_interval == 0:
             if new_frame_interval != 0:
                 same_config = False
        elif new_frame_interval % self.frame_interval != 0 or new_threshold < self.threshold:
            same_config = False

        if same_config:
            return model_toReturn
        else:
            return 2

    def __str__(self) -> str:
        return f"ModelInfo(frame_interval={self.frame_interval}, threshold={self.threshold}, version={self.version}, ai_model_id={self.ai_model_id}, file_name={self.file_name})"

class VideoMetadata(BaseModel):
    duration: float
    models: Dict[str, ModelInfo]
    def __str__(self) -> str:
        return f"VideoMetadata(duration={self.duration}, models={self.models})"


class AIVideoResult(BaseModel):
    schema_version: int
    metadata: VideoMetadata
    timespans: Dict[str, Dict[str, List[TagTimeFrame]]]

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)

    def add_server_result(self, server_result: Dict[str, Any]) -> None:
        ai_version_and_ids: List[Tuple[float, int, Optional[str], List[str]]] = server_result['ai_models_info']
        updated_categories: Set[str] = set()
        current_models: Dict[str, ModelInfo] = self.metadata.models

        frame_interval: float = float(server_result['frame_interval'])
        threshold: float = float(server_result['threshold'])
        
        ai_version: float
        ai_id: int
        ai_filename: Optional[str]
        ai_categories: List[str]
        category: str
        
        for ai_version, ai_id, ai_filename, ai_categories in ai_version_and_ids:
            for category in ai_categories:
                if category in current_models:
                    model_info: ModelInfo = current_models[category]
                    if model_info.needs_reprocessed(frame_interval, threshold, ai_version, ai_id, ai_filename) > 0:
                        current_models[category] = ModelInfo(frame_interval=frame_interval, threshold=threshold, version=ai_version, ai_model_id=ai_id, file_name=ai_filename)
                        updated_categories.add(category)
                else:
                    current_models[category] = ModelInfo(frame_interval=frame_interval, threshold=threshold, version=ai_version, ai_model_id=ai_id, file_name=ai_filename)
                    updated_categories.add(category)

        frames: List[Dict[str, Any]] = server_result['frames']
        processed_timespans: Dict[str, Dict[str, List[TagTimeFrame]]] = AIVideoResult.__mutate_server_result_tags(frames, frame_interval)
        logger.debug(f"Updated categories: {updated_categories}")
        for category in updated_categories:
            if category in processed_timespans:
                self.timespans[category] = processed_timespans[category]
            else:
                logger.warning(f"Category {category} updated in models but no new timespan data found. Clearing existing timespans for this category.")
                self.timespans[category] = {}
    
    @classmethod
    def from_client_json(cls, json_data: Optional[Dict[str, Any]]) -> Tuple[Optional['AIVideoResult'], bool]:
        if json_data is None:
            return None, True
        if "schema_version" not in json_data:
            from lib.model.postprocessing.AI_VideoResultV0 import AIVideoResultV0
            v0: 'AIVideoResultV0' = AIVideoResultV0(**json_data)
            v1_result: 'AIVideoResult' = v0.to_V1()
            return v1_result, True
        else:
            return cls(**json_data), False

    @classmethod
    def from_server_result(cls, server_result: Dict[str, Any]) -> 'AIVideoResult':
        frames: List[Dict[str, Any]] = server_result['frames']
        video_duration: float = float(server_result['video_duration'])
        frame_interval: float = float(server_result['frame_interval'])
        
        timespans: Dict[str, Dict[str, List[TagTimeFrame]]] = AIVideoResult.__mutate_server_result_tags(frames, frame_interval)
        
        ai_version_and_ids: List[Tuple[float, int, Optional[str], List[str]]] = server_result['ai_models_info']
        modelinfos: Dict[str, ModelInfo] = {}
        
        ai_version: float
        ai_id: int
        ai_filename: Optional[str]
        ai_categories: List[str]
        category: str
        
        for ai_version, ai_id, ai_filename, ai_categories in ai_version_and_ids:
            model_info_params = {
                "frame_interval": frame_interval,
                "threshold": float(server_result['threshold']),
                "version": ai_version,
                "ai_model_id": ai_id,
                "file_name": ai_filename
            }
            model_info: ModelInfo = ModelInfo(**model_info_params)
            for category in ai_categories:
                if category in modelinfos:
                    logger.error(f"Category {category} already exists in modelinfos. Models may have overlapping categories. Overwriting.")
                modelinfos[category] = model_info
                
        metadata: VideoMetadata = VideoMetadata(duration=video_duration, models=modelinfos)
        schema_version: int = 1
        return cls(schema_version=schema_version, metadata=metadata, timespans=timespans)

    @classmethod
    def __mutate_server_result_tags(cls, frames: List[Dict[str, Any]], frame_interval: float) -> Dict[str, Dict[str, List[TagTimeFrame]]]:
        toReturn: Dict[str, Dict[str, List[TagTimeFrame]]] = {}
        
        frame_data: Dict[str, Any]
        for frame_data in frames:
            frame_index: float = float(frame_data['frame_index'])
            
            key: str 
            value: Any
            for key, value in frame_data.items():
                if key != "frame_index":
                    currentCategoryDict: Dict[str, List[TagTimeFrame]]
                    if key in toReturn:
                        currentCategoryDict = toReturn[key]
                    else:
                        currentCategoryDict = {}
                        toReturn[key] = currentCategoryDict
                    
                    if not isinstance(value, list):
                        logger.warning(f"Category data for '{key}' is not a list, skipping. Got: {type(value)}")
                        continue

                    item_data: Any
                    for item_data in value:
                        tag_name: str
                        confidence: Optional[float]
                        
                        if isinstance(item_data, tuple) and len(item_data) == 2:
                            tag_name = str(item_data[0])
                            confidence = float(item_data[1]) if item_data[1] is not None else None
                        elif isinstance(item_data, str):
                            tag_name = item_data
                            confidence = None
                        else:
                            logger.warning(f"Skipping unrecognized item format in category '{key}': {item_data}")
                            continue

                        if tag_name not in currentCategoryDict:
                            currentCategoryDict[tag_name] = [TagTimeFrame(start=frame_index, end=None, confidence=confidence)]
                        else:
                            last_time_frame: TagTimeFrame = currentCategoryDict[tag_name][-1]

                            if last_time_frame.end is None:
                                if (frame_index - float(last_time_frame.start)) == frame_interval and last_time_frame.confidence == confidence:
                                    last_time_frame.end = frame_index
                                else:
                                    currentCategoryDict[tag_name].append(TagTimeFrame(start=frame_index, end=None, confidence=confidence))
                            elif last_time_frame.confidence == confidence and (frame_index - float(last_time_frame.end)) == frame_interval:
                                last_time_frame.end = frame_index
                            else:
                                currentCategoryDict[tag_name].append(TagTimeFrame(start=frame_index, end=None, confidence=confidence))
        return toReturn

def save_gzip_json(data: 'AIVideoResult', file_path: str) -> bool:
    try:
        json_str = data.to_json()
        gzipped_json = gzip.compress(json_str.encode('utf-8'))
        with open(file_path, 'wb') as f:
            f.write(gzipped_json)
        return True
    except Exception as e:
        logger.error(f"Error saving gzipped JSON to {file_path}: {e}", exc_info=True)
        return False

def load_gzip_json(file_path: str) -> Optional['AIVideoResult']:
    try:
        with open(file_path, 'rb') as f:
            gzipped_json = f.read()
        json_str = gzip.decompress(gzipped_json).decode('utf-8')
        data = json.loads(json_str)
        return AIVideoResult.from_client_json(data)[0]
    except FileNotFoundError:
        logger.debug(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading gzipped JSON from {file_path}: {e}", exc_info=True)
        return None