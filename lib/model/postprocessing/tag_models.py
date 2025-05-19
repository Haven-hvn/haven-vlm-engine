from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, field_validator, ConfigDict
import math
import logging

model_logger = logging.getLogger(__name__)

class TimeFrame(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    start: int
    end: int
    totalConfidence: Optional[float]

    @field_validator('start', 'end', mode='before')
    @classmethod
    def _ensure_floored_int(cls, v: Any) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            # model_logger.debug(f"Validator converting float {v} to int {int(math.floor(v))}")
            return int(math.floor(v))
        try:
            # model_logger.debug(f"Validator attempting to convert value {v} (type {type(v)}) to int")
            return int(math.floor(float(str(v))))
        except (ValueError, TypeError) as e:
            model_logger.error(f"TimeFrame field validator: Could not convert value '{v}' (type {type(v)}) to int. Error: {e}")
            raise ValueError(f"Invalid value '{v}' for start/end, cannot convert to integer.") from e

    def get_density(self, frame_interval: float) -> float:
        duration: float = self.get_duration(frame_interval)
        if duration == 0:
            return 0.0
        if self.totalConfidence is None:
            return 0.0
        return self.totalConfidence / duration
    
    def get_duration(self, frame_interval: float) -> float:
        return (self.end - self.start) + frame_interval # Direct field access
    
    def merge(self, new_start: float, new_end: float, new_confidence: float, frame_interval: float) -> None:
        # Assignment to self.start and self.end will trigger the validator 
        # due to validate_assignment=True and mode='before' on the validator.
        self.start = min(self.start, new_start) 
        self.end = max(self.end, new_end)
        
        if self.totalConfidence is None:
            self.totalConfidence = 0.0
        self.totalConfidence += new_confidence * self.get_duration(frame_interval)

    def __str__(self) -> str:
        return f"TimeFrame(start={self.start}, end={self.end}, totalConfidence={self.totalConfidence})"
    
class VideoTagInfo(BaseModel):
    video_duration: float
    video_tags: Dict[str, Set[str]]
    tag_totals: Dict[str, Dict[str, float]]
    tag_timespans: Dict[str, Dict[str, List[TimeFrame]]]

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)

    def __str__(self) -> str:
        return f"VideoTagInfo(video_duration={self.video_duration}, video_tags={self.video_tags}, tag_totals={self.tag_totals}, tag_timespans={self.tag_timespans})"