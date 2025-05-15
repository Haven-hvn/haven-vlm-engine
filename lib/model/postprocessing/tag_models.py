from typing import Dict, List, Optional, Set
from pydantic import BaseModel

class TimeFrame(BaseModel):
    start: float
    end: float
    totalConfidence: Optional[float]

    def get_density(self, frame_interval: float) -> float:
        # Avoid division by zero if duration can be zero
        duration: float = self.get_duration(frame_interval)
        if duration == 0:
            return 0.0 # Or handle as an error, depending on expected behavior
        # Ensure totalConfidence is not None before division
        if self.totalConfidence is None:
            return 0.0 # Or raise error
        return self.totalConfidence / duration
    
    def get_duration(self, frame_interval: float) -> float:
        return (self.end - self.start) + frame_interval
    
    def merge(self, new_start: float, new_end: float, new_confidence: float, frame_interval: float) -> None:
        self.start = min(self.start, new_start)
        self.end = max(self.end, new_end)
        # Ensure totalConfidence is not None before operation
        if self.totalConfidence is None:
            self.totalConfidence = 0.0 # Initialize if None
        # Calculate duration of the new segment to be merged, effectively.
        # This logic seems to weight new_confidence by the existing duration + frame_interval
        # which might not be standard if merging two segments. 
        # Assuming this custom logic is intended.
        self.totalConfidence += new_confidence * self.get_duration(frame_interval) # Original logic
        # Alternative if merging segments: self.totalConfidence += new_confidence * (new_end - new_start + frame_interval)

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