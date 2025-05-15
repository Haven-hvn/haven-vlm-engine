from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from lib.model.postprocessing import tag_models

class ImagePathList(BaseModel):
    paths: List[str]
    pipeline_name: Optional[str] = None
    threshold: Optional[float] = None
    return_confidence: Optional[bool] = None

class VideoPathList(BaseModel):
    path: str
    returnTimestamps: bool = True
    pipeline_name: Optional[str] = None
    frame_interval: Optional[float] = None
    threshold: Optional[float] = None
    return_confidence: Optional[bool] = None
    vr_video: bool = False
    existing_json_data: Optional[Any] = None

class OptimizeMarkerSettings(BaseModel):
    existing_json_data: Optional[Any] = None
    desired_timespan_data: Optional[Any] = None

class VideoResult(BaseModel):
    result: Any

class ImageResult(BaseModel):
    result: Any