import logging
import time
from lib.async_lib.async_processing import ItemFuture, QueueItem
from lib.model.model import Model
from lib.model.preprocessing_python.image_preprocessing import preprocess_video
from typing import Dict, Any, List, Optional, Union, Tuple

class VideoPreprocessorModel(Model):
    def __init__(self, configValues: Dict[str, Any]):
        super().__init__(configValues)
        self.image_size: Union[int, Tuple[int, int]] = configValues.get("image_size", 512)
        self.frame_interval: float = float(configValues.get("frame_interval", 0.5))
        self.use_half_precision: bool = bool(configValues.get("use_half_precision", True))
        self.device: Optional[str] = configValues.get("device")
        self.normalization_config: Union[int, Dict[str, List[float]]] = configValues.get("normalization_config", 1)
        self.logger: logging.Logger = logging.getLogger("logger")
    
    async def worker_function(self, queue_items: List[QueueItem]) -> None:
        item: QueueItem
        for item in queue_items:
            itemFuture: ItemFuture = item.item_future
            try:
                totalTime: float = 0.0
                video_path: str = itemFuture[item.input_names[0]]
                use_timestamps: bool = itemFuture[item.input_names[1]]
                frame_interval_override: Optional[float] = itemFuture.get(item.input_names[2])
                current_frame_interval: float = frame_interval_override if frame_interval_override is not None else self.frame_interval
                vr_video: bool = itemFuture[item.input_names[5]]
                
                children: List[ItemFuture] = []
                processed_frames_count: int = 0
                oldTime: float = time.time()
                norm_config_to_use: Union[int, Dict[str, List[float]]] = self.normalization_config
                
                frame_index: int
                frame_tensor: Any
                for frame_index, frame_tensor in preprocess_video(video_path, current_frame_interval, self.image_size, self.use_half_precision, self.device, use_timestamps, vr_video=vr_video, norm_config=norm_config_to_use):
                    processed_frames_count += 1
                    newTime: float = time.time()
                    totalTime += newTime - oldTime
                    oldTime = newTime
                    
                    future_data_payload: Dict[str, Any] = {
                        item.output_names[1]: frame_tensor, 
                        item.output_names[2]: frame_index,
                        item.output_names[3]: itemFuture.get(item.input_names[3]),
                        item.output_names[4]: itemFuture.get(item.input_names[4]),
                        item.output_names[5]: itemFuture.get(item.input_names[6])
                    }
                    result_future: ItemFuture = await ItemFuture.create(item, future_data_payload, item.item_future.handler)
                    children.append(result_future)
                
                if processed_frames_count > 0:
                    self.logger.info(f"Preprocessed {processed_frames_count} frames in {totalTime:.2f} seconds at an average of {totalTime/processed_frames_count:.3f} seconds per frame.")
                else:
                    self.logger.info(f"No frames preprocessed for {video_path}.")
                
                await itemFuture.set_data(item.output_names[0], children)
            except FileNotFoundError as fnf_error:
                self.logger.error(f"File not found error processing {itemFuture.get(item.input_names[0], 'unknown_file')}: {fnf_error}")
                itemFuture.set_exception(fnf_error)
            except IOError as io_error:
                self.logger.error(f"IO error (video might be corrupted) processing {itemFuture.get(item.input_names[0], 'unknown_file')}: {io_error}")
                itemFuture.set_exception(io_error)
            except Exception as e:
                self.logger.error(f"An unexpected error occurred processing {itemFuture.get(item.input_names[0], 'unknown_file')}: {e}", exc_info=True)
                itemFuture.set_exception(e)