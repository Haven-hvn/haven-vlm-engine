from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo
import lib.model.postprocessing.AI_VideoResult as AI_VideoResult
from lib.model.postprocessing.category_settings import category_config
import math
import logging

tag_to_category_dict: Dict[str, str] = {}

category: str
tags: Dict[str, Dict[str, str]] # Based on category_config structure
tag: str
for category, tags in category_config.items():
    for tag in tags:
        tag_to_category_dict[tag] = category

class ModelConfigV0(BaseModel):
    frame_interval: float
    threshold: float
    def __str__(self) -> str:
        return f"ModelConfig(frame_interval={self.frame_interval}, threshold={self.threshold})"
    
class ModelInfoV0(BaseModel):
    version: float
    ai_model_config: ModelConfigV0
    def __str__(self) -> str:
        return f"ModelInfo(version={self.version}, ai_model_config={self.ai_model_config})"
    
class VideoMetadataV0(BaseModel):
    video_id: int
    duration: float
    phash: Optional[str]
    models: Dict[str, ModelInfoV0]
    frame_interval: float
    threshold: float
    ai_version: float
    ai_model_id: int
    ai_model_filename: Optional[str] = None
    def __str__(self) -> str:
        return f"VideoMetadata(video_id={self.video_id}, duration={self.duration}, phash={self.phash}, models={self.models})"
    
class TagTimeFrameV0(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    start: int
    end: Optional[int] = None
    confidence: float

    @field_validator('start', 'end', mode='before')
    @classmethod
    def _floor_and_convert_to_int_v0(cls, v: Any, info: ValidationInfo) -> Optional[int]:
        if v is None and info.field_name == 'end':
            return None
        if v is None:
            raise ValueError(f"TagTimeFrameV0 field '{info.field_name}' received None but is not Optional or handled.")

        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(math.floor(v))
        try:
            return int(math.floor(float(str(v))))
        except (ValueError, TypeError) as e:
            logging.error(f"TagTimeFrameV0 validator: Could not convert '{v}' for '{info.field_name}' to int: {e}")
            raise ValueError(f"Invalid value '{v}' for TagTimeFrameV0 field '{info.field_name}', cannot convert to int.") from e

    def __str__(self) -> str:
        return f"TagTimeFrame(start={self.start}, end={self.end}, confidence={self.confidence})"
    
class TagDataV0(BaseModel):
    ai_model_name: str
    time_frames: List[TagTimeFrameV0]
    def __str__(self) -> str:
        return f"TagData(model_name={self.ai_model_name}, time_frames={self.time_frames})"

class AIVideoResultV0(BaseModel):
    video_metadata: VideoMetadataV0
    tags: Dict[str, TagDataV0]
    timespans: Dict[str, Dict[str, List[TagTimeFrameV0]]]

    def to_V1(self) -> AI_VideoResult.AIVideoResult:
        from lib.model.postprocessing.AI_VideoResult import AIVideoResult, VideoMetadata, ModelInfo

        model_infos_v1: Dict[str, ModelInfo] = {}
        if self.timespans:
            shared_model_info = ModelInfo(
                frame_interval=self.video_metadata.frame_interval,
                threshold=self.video_metadata.threshold,
                version=self.video_metadata.ai_version,
                ai_model_id=self.video_metadata.ai_model_id,
                file_name=self.video_metadata.ai_model_filename
            )
            for category_name in self.timespans.keys():
                model_infos_v1[category_name] = shared_model_info

        metadata_v1 = VideoMetadata(duration=self.video_metadata.duration, models=model_infos_v1)
        
        v1_timespans: Dict[str, Dict[str, List[AI_VideoResult.TagTimeFrame]]] = {}
        for category, tags_map in self.timespans.items():
            v1_tags_map: Dict[str, List[AI_VideoResult.TagTimeFrame]] = {}
            for tag_name, ttv0_list in tags_map.items():
                v1_tags_map[tag_name] = [AI_VideoResult.TagTimeFrame(start=ttv0.start, end=ttv0.end, confidence=ttv0.confidence) for ttv0 in ttv0_list]
            v1_timespans[category] = v1_tags_map

        return AIVideoResult(
            schema_version=1,
            metadata=metadata_v1,
            timespans=v1_timespans
        )