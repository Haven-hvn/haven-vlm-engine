import logging
import time
from lib.async_lib.async_processing import ItemFuture, QueueItem
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import preprocess_video
from typing import Dict, Any, List, Optional, Union, Tuple
from .performance_metrics import PerformanceTracker

class VideoPreprocessorModel(Model):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.logger = logging.getLogger("logger")
        self.device: str = model_config.get("device", "cpu")
        self.image_size: Union[int, List[int]] = model_config.get("image_size", 512)
        self.frame_interval: float = model_config.get("frame_interval", 0.5)
        self.use_half_precision: bool = model_config.get("use_half_precision", True)
        self.normalization_config: Union[int, Dict[str, List[float]]] = model_config.get("normalization_config", 1)
        self.process_for_vlm: bool = False
        self.performance_tracker = PerformanceTracker()
    
    def set_vlm_pipeline_mode(self, mode: bool) -> None:
        self.process_for_vlm = mode
        self.logger.info(f"VideoPreprocessorModel VLM mode set to: {self.process_for_vlm}")

    async def worker_function(self, queue_items: List[QueueItem]) -> None:
        item: QueueItem
        for item in queue_items:
            itemFuture: ItemFuture = item.item_future
            try:
                video_path: str = itemFuture[item.input_names[0]]
                self.performance_tracker.start_processing(video_path)
                self.performance_tracker.start_preprocessing()
                
                use_timestamps: bool = itemFuture[item.input_names[1]]
                frame_interval_override: Optional[float] = itemFuture[item.input_names[2]]
                current_frame_interval: float = frame_interval_override if frame_interval_override is not None else self.frame_interval
                vr_video: bool = itemFuture[item.input_names[5]]
                
                children: List[ItemFuture] = []
                processed_frames_count: int = 0
                norm_config_to_use: Union[int, Dict[str, List[float]]] = self.normalization_config
                
                frame_index: int
                frame_tensor: Any
                for frame_index, frame_tensor in preprocess_video(video_path, current_frame_interval, self.image_size, self.use_half_precision, self.device, use_timestamps, vr_video=vr_video, norm_config_idx=norm_config_to_use, process_for_vlm=self.process_for_vlm):
                    processed_frames_count += 1
                    self.performance_tracker.increment_frames()
                    
                    future_data_payload: Dict[str, Any] = {
                        item.output_names[1]: frame_tensor, 
                        item.output_names[2]: frame_index,
                        item.output_names[3]: itemFuture[item.input_names[3]],
                        item.output_names[4]: itemFuture[item.input_names[4]],
                        item.output_names[5]: itemFuture[item.input_names[6]]
                    }
                    result_future: ItemFuture = await ItemFuture.create(item, future_data_payload, item.item_future.handler)
                    children.append(result_future)
                
                self.performance_tracker.end_preprocessing()
                
                if processed_frames_count > 0:
                    metrics = self.performance_tracker.get_metrics()
                    self.logger.info(
                        f"Preprocessed {processed_frames_count} frames in {metrics.preprocessing_time:.2f} seconds "
                        f"at an average of {metrics.preprocessing_fps:.2f} FPS. "
                        f"Total processing time: {metrics.total_processing_time:.2f} seconds."
                    )
                else:
                    self.logger.info(f"No frames preprocessed for {video_path}.")
                
                await itemFuture.set_data(item.output_names[0], children)
            except FileNotFoundError as fnf_error:
                video_file_path_for_log = itemFuture[item.input_names[0]] or 'unknown_file'
                self.logger.error(f"File not found error processing {video_file_path_for_log}: {fnf_error}")
                itemFuture.set_exception(fnf_error)
            except IOError as io_error:
                video_file_path_for_log = itemFuture[item.input_names[0]] or 'unknown_file'
                self.logger.error(f"IO error (video might be corrupted) processing {video_file_path_for_log}: {io_error}")
                itemFuture.set_exception(io_error)
            except Exception as e:
                video_file_path_for_log = itemFuture[item.input_names[0]] or 'unknown_file'
                self.logger.error(f"An unexpected error occurred processing {video_file_path_for_log}: {e}", exc_info=True)
                itemFuture.set_exception(e)