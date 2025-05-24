from dataclasses import dataclass
from typing import Dict, Optional, List
import time
from datetime import datetime

@dataclass
class VideoProcessingMetrics:
    total_frames: int
    total_processing_time: float
    preprocessing_time: float
    model_inference_time: float
    postprocessing_time: float
    start_time: datetime
    end_time: datetime
    
    @property
    def average_fps(self) -> float:
        """Calculate average frames per second across the entire processing pipeline."""
        return self.total_frames / self.total_processing_time if self.total_processing_time > 0 else 0.0
    
    @property
    def preprocessing_fps(self) -> float:
        """Calculate frames per second for the preprocessing stage."""
        return self.total_frames / self.preprocessing_time if self.preprocessing_time > 0 else 0.0
    
    @property
    def model_inference_fps(self) -> float:
        """Calculate frames per second for the model inference stage."""
        return self.total_frames / self.model_inference_time if self.model_inference_time > 0 else 0.0
    
    @property
    def postprocessing_fps(self) -> float:
        """Calculate frames per second for the postprocessing stage."""
        return self.total_frames / self.postprocessing_time if self.postprocessing_time > 0 else 0.0

@dataclass
class PipelineMetrics:
    """Aggregates metrics from all pipeline stages."""
    video_path: str
    total_frames: int
    total_pipeline_time: float
    preprocessing_metrics: VideoProcessingMetrics
    model_inference_metrics: VideoProcessingMetrics
    postprocessing_metrics: VideoProcessingMetrics
    start_time: datetime
    end_time: datetime
    
    @property
    def overall_fps(self) -> float:
        """Calculate overall frames per second across the entire pipeline."""
        return self.total_frames / self.total_pipeline_time if self.total_pipeline_time > 0 else 0.0
    
    @property
    def stage_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get a breakdown of metrics for each pipeline stage."""
        return {
            "preprocessing": {
                "time": self.preprocessing_metrics.preprocessing_time,
                "fps": self.preprocessing_metrics.preprocessing_fps,
                "percentage": (self.preprocessing_metrics.preprocessing_time / self.total_pipeline_time * 100) if self.total_pipeline_time > 0 else 0.0
            },
            "model_inference": {
                "time": self.model_inference_metrics.model_inference_time,
                "fps": self.model_inference_metrics.model_inference_fps,
                "percentage": (self.model_inference_metrics.model_inference_time / self.total_pipeline_time * 100) if self.total_pipeline_time > 0 else 0.0
            },
            "postprocessing": {
                "time": self.postprocessing_metrics.postprocessing_time,
                "fps": self.postprocessing_metrics.postprocessing_fps,
                "percentage": (self.postprocessing_metrics.postprocessing_time / self.total_pipeline_time * 100) if self.total_pipeline_time > 0 else 0.0
            }
        }
    
    def to_log_string(self) -> str:
        """Convert metrics to a formatted log string."""
        breakdown = self.stage_breakdown
        return (
            f"\nPipeline Performance Metrics for {self.video_path}:"
            f"\nTotal Frames: {self.total_frames}"
            f"\nTotal Pipeline Time: {self.total_pipeline_time:.2f}s"
            f"\nOverall FPS: {self.overall_fps:.2f}"
            f"\nStage Breakdown:"
            f"\n  Preprocessing:"
            f"\n    Time: {breakdown['preprocessing']['time']:.2f}s ({breakdown['preprocessing']['percentage']:.1f}%)"
            f"\n    FPS: {breakdown['preprocessing']['fps']:.2f}"
            f"\n  Model Inference:"
            f"\n    Time: {breakdown['model_inference']['time']:.2f}s ({breakdown['model_inference']['percentage']:.1f}%)"
            f"\n    FPS: {breakdown['model_inference']['fps']:.2f}"
            f"\n  Postprocessing:"
            f"\n    Time: {breakdown['postprocessing']['time']:.2f}s ({breakdown['postprocessing']['percentage']:.1f}%)"
            f"\n    FPS: {breakdown['postprocessing']['fps']:.2f}"
        )

class PerformanceTracker:
    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.preprocessing_start: Optional[float] = None
        self.model_inference_start: Optional[float] = None
        self.postprocessing_start: Optional[float] = None
        self.total_frames: int = 0
        self.preprocessing_time: float = 0.0
        self.model_inference_time: float = 0.0
        self.postprocessing_time: float = 0.0
        self.video_path: Optional[str] = None
    
    def start_processing(self, video_path: Optional[str] = None) -> None:
        """Start tracking overall processing time."""
        self.start_time = datetime.now()
        if video_path:
            self.video_path = video_path
    
    def start_preprocessing(self) -> None:
        """Start tracking preprocessing time."""
        self.preprocessing_start = time.time()
    
    def end_preprocessing(self) -> None:
        """End tracking preprocessing time."""
        if self.preprocessing_start is not None:
            self.preprocessing_time = time.time() - self.preprocessing_start
    
    def start_model_inference(self) -> None:
        """Start tracking model inference time."""
        self.model_inference_start = time.time()
    
    def end_model_inference(self) -> None:
        """End tracking model inference time."""
        if self.model_inference_start is not None:
            self.model_inference_time = time.time() - self.model_inference_start
    
    def start_postprocessing(self) -> None:
        """Start tracking postprocessing time."""
        self.postprocessing_start = time.time()
    
    def end_postprocessing(self) -> None:
        """End tracking postprocessing time."""
        if self.postprocessing_start is not None:
            self.postprocessing_time = time.time() - self.postprocessing_start
    
    def increment_frames(self, count: int = 1) -> None:
        """Increment the total frame count."""
        self.total_frames += count
    
    def get_metrics(self) -> VideoProcessingMetrics:
        """Get the final processing metrics."""
        if self.start_time is None:
            raise ValueError("Processing was not started")
        
        end_time = datetime.now()
        total_processing_time = (end_time - self.start_time).total_seconds()
        
        return VideoProcessingMetrics(
            total_frames=self.total_frames,
            total_processing_time=total_processing_time,
            preprocessing_time=self.preprocessing_time,
            model_inference_time=self.model_inference_time,
            postprocessing_time=self.postprocessing_time,
            start_time=self.start_time,
            end_time=end_time
        )
    
    def get_pipeline_metrics(self, preprocessing_metrics: VideoProcessingMetrics, 
                           model_inference_metrics: VideoProcessingMetrics,
                           postprocessing_metrics: VideoProcessingMetrics) -> PipelineMetrics:
        """Get aggregated metrics from all pipeline stages."""
        if self.start_time is None:
            raise ValueError("Processing was not started")
        
        end_time = datetime.now()
        total_pipeline_time = (end_time - self.start_time).total_seconds()
        
        return PipelineMetrics(
            video_path=self.video_path or "unknown",
            total_frames=self.total_frames,
            total_pipeline_time=total_pipeline_time,
            preprocessing_metrics=preprocessing_metrics,
            model_inference_metrics=model_inference_metrics,
            postprocessing_metrics=postprocessing_metrics,
            start_time=self.start_time,
            end_time=end_time
        ) 