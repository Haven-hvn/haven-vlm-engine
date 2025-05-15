from typing import Dict, List, Optional
from pydantic import BaseModel
import lib.model.postprocessing.AI_VideoResult as AI_VideoResult
from lib.model.postprocessing.category_settings import category_config

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
    def __str__(self) -> str:
        return f"VideoMetadata(video_id={self.video_id}, duration={self.duration}, phash={self.phash}, models={self.models})"
    
class TagTimeFrameV0(BaseModel):
    start: float
    end: Optional[float] = None
    confidence: float
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

    def to_V1(self) -> AI_VideoResult.AIVideoResult:
        models: Dict[str, AI_VideoResult.ModelInfo] = {}
        model_name: str
        model_info: ModelInfoV0
        for model_name, model_info in self.video_metadata.models.items():
            if model_name == "actiondetection":
                if model_info.version == 2.0:
                    models["actions"] = AI_VideoResult.ModelInfo(frame_interval=model_info.ai_model_config.frame_interval, threshold=model_info.ai_model_config.threshold, version=2.0, ai_model_id=200, file_name="gentler_river")
                else:
                    models["actions"] = AI_VideoResult.ModelInfo(frame_interval=model_info.ai_model_config.frame_interval, threshold=model_info.ai_model_config.threshold, version=0.1, ai_model_id=950)
            elif model_name == "bodyparts":
                models["bodyparts"] = AI_VideoResult.ModelInfo(frame_interval=model_info.ai_model_config.frame_interval, threshold=model_info.ai_model_config.threshold, version=0.5, ai_model_id=200, file_name="iconic_sky")
            elif model_name == "actiondetection_bodyparts_bdsm":
                models["bdsm"] = AI_VideoResult.ModelInfo(frame_interval=model_info.ai_model_config.frame_interval, threshold=model_info.ai_model_config.threshold, version=0.7, ai_model_id=200, file_name="true_lake")
                models["actions"] = AI_VideoResult.ModelInfo(frame_interval=model_info.ai_model_config.frame_interval, threshold=model_info.ai_model_config.threshold, version=2.0, ai_model_id=200, file_name="gentler_river")
                models["bodyparts"] = AI_VideoResult.ModelInfo(frame_interval=model_info.ai_model_config.frame_interval, threshold=model_info.ai_model_config.threshold, version=0.5, ai_model_id=200, file_name="iconic_sky")
            elif model_name == "actiondetection_bodyparts":
                models["actions"] = AI_VideoResult.ModelInfo(frame_interval=model_info.ai_model_config.frame_interval, threshold=model_info.ai_model_config.threshold, version=2.0, ai_model_id=200, file_name="gentler_river")
                models["bodyparts"] = AI_VideoResult.ModelInfo(frame_interval=model_info.ai_model_config.frame_interval, threshold=model_info.ai_model_config.threshold, version=0.5, ai_model_id=200, file_name="iconic_sky")
                    

        metadata: AI_VideoResult.VideoMetadata = AI_VideoResult.VideoMetadata(duration=self.video_metadata.duration, models=models)
        timespans: Dict[str, Dict[str, List[AI_VideoResult.TagTimeFrame]]] = {}
        tag: str
        tag_data: TagDataV0
        for tag, tag_data in self.tags.items():
            category: str = tag_to_category_dict.get(tag, "Unknown")
            if category not in timespans:
                timespans[category] = {}
            time_frame: TagTimeFrameV0
            timespans[category][tag] = [AI_VideoResult.TagTimeFrame(start=time_frame.start, end=time_frame.end, confidence=time_frame.confidence) for time_frame in tag_data.time_frames]
        return AI_VideoResult.AIVideoResult(schema_version=1, metadata=metadata, timespans=timespans)