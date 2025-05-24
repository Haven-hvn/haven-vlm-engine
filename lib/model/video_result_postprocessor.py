from typing import Any, Dict, List, Optional, Union
import logging
from .performance_metrics import PerformanceTracker

class VideoResultPostprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_tracker = PerformanceTracker()

    async def process(self, item: Any, itemFuture: Any) -> None:
        try:
            self.performance_tracker.start_postprocessing()
            
            # Process the video results
            children_results = itemFuture[item.input_names[0]]
            video_path = itemFuture[item.input_names[1]]
            time_interval = itemFuture[item.input_names[2]]
            threshold = itemFuture[item.input_names[3]]
            existing_video_data = itemFuture[item.input_names[4]]
            
            # Process results and update metrics
            self.performance_tracker.increment_frames(len(children_results))
            
            self.performance_tracker.end_postprocessing()
            metrics = self.performance_tracker.get_metrics()
            
            # Get pipeline metrics from all stages
            pipeline_metrics = self.performance_tracker.get_pipeline_metrics(
                preprocessing_metrics=metrics,
                model_inference_metrics=metrics,
                postprocessing_metrics=metrics
            )
            
            # Log detailed pipeline metrics
            self.logger.info(pipeline_metrics.to_log_string())
            
            # Set the processed results
            await itemFuture.set_data(item.output_names[0], {
                "results": children_results,
                "video_path": video_path,
                "time_interval": time_interval,
                "threshold": threshold,
                "existing_video_data": existing_video_data,
                "performance_metrics": pipeline_metrics
            })
            
        except Exception as e:
            self.logger.error(f"Error in video result postprocessing: {str(e)}")
            itemFuture.set_exception(e) 