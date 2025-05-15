import asyncio
import logging
import os
from PIL import Image
from lib.config.config_utils import load_config
from lib.model.preprocessing_python.image_preprocessing import get_video_duration_decord
from lib.model.postprocessing.AI_VideoResult import AIVideoResult
import lib.model.postprocessing.timeframe_processing as timeframe_processing
from lib.model.postprocessing.category_settings import category_config
from lib.model.skip_input import Skip
from lib.model.postprocessing.post_processing_settings import get_or_default, post_processing_config
from lib.model.vlm_model import OpenAICompatibleVLMClient
from typing import List, Dict, Any, Optional, Union, Tuple
from lib.async_lib.async_processing import ItemFuture, QueueItem
import numpy as np

logger: logging.Logger = logging.getLogger("logger")

# Global VLM model instance, assuming OpenAICompatibleVLMClient is the one intended
_vlm_model: Optional[OpenAICompatibleVLMClient] = None

def get_vlm_model(config: Dict[str, Any]) -> OpenAICompatibleVLMClient:
    """Get or create the VLM model instance"""
    global _vlm_model
    if _vlm_model is None:
        logger.info(f"Initializing VLM model with config: {config.get('model_id')}")
        # Pass the full config dictionary to the client
        _vlm_model = OpenAICompatibleVLMClient(config=config)
    return _vlm_model

async def vlm_frame_analyzer(data: List[QueueItem]) -> None:
    """
    Process a frame using the VLM model.
    Returns a dictionary of {tag: confidence} for each tag in the tag list.
    Assumes data contains QueueItems with necessary future and input_names.
    Input anmes should be [image_tensor, client_config_for_vlm]
    """
    item: QueueItem
    for item in data:
        item_future: ItemFuture = item.item_future
        try:
            # Assuming input_names[0] is frame_tensor, input_names[1] is client_config
            frame_input: Any = item_future[item.input_names[0]] # Can be PIL.Image or tensor-like
            client_config: Dict[str, Any] = item_future[item.input_names[1]]
            
            frame_pil: Image.Image
            if isinstance(frame_input, Image.Image):
                frame_pil = frame_input
            elif hasattr(frame_input, 'cpu') and hasattr(frame_input, 'numpy'): # Basic check for tensor-like
                # Convert tensor to PIL Image (basic example, might need more complex preprocessing)
                # This assumes CHW format and normalizes to [0, 255] uint8
                img_np = frame_input.cpu().numpy()
                if img_np.ndim == 3 and img_np.shape[0] == 3: # CHW
                    img_np = np.transpose(img_np, (1,2,0)) # HWC
                if img_np.dtype != np.uint8:
                     # Attempt to scale if float and in [0,1] range, otherwise assume it's [0,255]
                    if (img_np.dtype == np.float32 or img_np.dtype == np.float64) and img_np.max() <=1.0 and img_np.min() >=0:
                        img_np = (img_np * 255)
                    img_np = img_np.astype(np.uint8)
                frame_pil = Image.fromarray(img_np)
            else:
                raise TypeError(f"Unsupported frame_input type: {type(frame_input)}")

            vlm: OpenAICompatibleVLMClient = get_vlm_model(client_config)
            scores: Dict[str, float] = vlm.analyze_frame(frame_pil)
            
            await item_future.set_data(item.output_names[0] if isinstance(item.output_names, list) else item.output_names, scores)
        except Exception as e:
            logger.error(f"Error in VLM frame analyzer: {e}")
            logger.debug("Stack trace:", exc_info=True)
            item_future.set_exception(e)

async def result_coalescer(data: List[QueueItem]) -> None:
    item: QueueItem
    for item in data:
        itemFuture: ItemFuture = item.item_future
        result: Dict[str, Any] = {}
        input_name: str
        for input_name in item.input_names:
            if input_name in itemFuture: # Check if key exists
                ai_result: Any = itemFuture[input_name]
                if not isinstance(ai_result, Skip):
                    result[input_name] = ai_result # No, use itemFuture[input_name] to get the actual value
        output_target = item.output_names[0] if isinstance(item.output_names, list) else item.output_names
        await itemFuture.set_data(output_target, result)
        
async def result_finisher(data: List[QueueItem]) -> None:
    item: QueueItem
    for item in data:
        itemFuture: ItemFuture = item.item_future
        # Ensure input_names[0] exists in itemFuture
        if item.input_names[0] in itemFuture:
            future_results: Any = itemFuture[item.input_names[0]]
            itemFuture.close_future(future_results)
        else:
            # Handle error: input not found
            logger.error(f"Input {item.input_names[0]} not found in itemFuture for result_finisher")
            itemFuture.set_exception(KeyError(f"Input {item.input_names[0]} not found"))

async def batch_awaiter(data: List[QueueItem]) -> None:
    item: QueueItem
    for item in data:
        itemFuture: ItemFuture = item.item_future
        if item.input_names[0] in itemFuture:
            futures: List[asyncio.Future] = itemFuture[item.input_names[0]]
            if isinstance(futures, list): # Ensure it's a list of futures
                results: List[Any] = await asyncio.gather(*futures, return_exceptions=True)
                output_target = item.output_names[0] if isinstance(item.output_names, list) else item.output_names
                await itemFuture.set_data(output_target, results)
            else:
                logger.error(f"Input for batch_awaiter ({item.input_names[0]}) is not a list of futures.")
                itemFuture.set_exception(TypeError("Input for batch_awaiter must be a list of futures."))
        else:
            logger.error(f"Input {item.input_names[0]} not found in itemFuture for batch_awaiter")
            itemFuture.set_exception(KeyError(f"Input {item.input_names[0]} not found"))

async def video_result_postprocessor(data: List[QueueItem]) -> None:
    item: QueueItem
    for item in data:
        itemFuture: ItemFuture = item.item_future
        # Assuming specific input names from a pipeline definition
        # Example: input_names = ["frame_results", "video_path", "frame_interval", "threshold", "video_result_object", "pipeline_ref"]
        try:
            duration: float = get_video_duration_decord(itemFuture[item.input_names[1]])
            result: Dict[str, Any] = {
                "frames": itemFuture[item.input_names[0]], 
                "video_duration": duration, 
                "frame_interval": float(itemFuture[item.input_names[2]]), 
                "threshold": float(itemFuture[item.input_names[3]]), 
                "ai_models_info": itemFuture['pipeline'].get_ai_models_info() # 'pipeline' should be passed in itemFuture
            }
            # It's safer not to delete from itemFuture.data directly unless managed carefully
            # del itemFuture.data["pipeline"] 

            videoResult: Optional[AIVideoResult] = itemFuture.get(item.input_names[4]) # Use .get for safety
            if videoResult is not None:
                videoResult.add_server_result(result)
            else:
                videoResult = AIVideoResult.from_server_result(result)

            toReturn: Dict[str, Any] = {"json_result": videoResult.to_json(), "video_tag_info": timeframe_processing.compute_video_tag_info(videoResult)}
            
            output_target = item.output_names[0] if isinstance(item.output_names, list) else item.output_names
            await itemFuture.set_data(output_target, toReturn)
        except Exception as e:
            logger.error(f"Error in video_result_postprocessor: {e}", exc_info=True)
            itemFuture.set_exception(e)

async def image_result_postprocessor(data: List[QueueItem]) -> None:
    item: QueueItem # Define item type for loop
    for item in data:
        itemFuture: ItemFuture = item.item_future
        toReturn: Dict[str, List[Union[Tuple[str, float], str]]] = {}
        try:
            # Assuming input_names[0] contains the raw image results from AI models
            # e.g., {"Category1": [("tagA", 0.9), "tagB"], "Category2": [...]} 
            raw_result: Dict[str, List[Union[Tuple[str, float], str]]] = itemFuture[item.input_names[0]]
            
            category: str
            tags_in_category: List[Union[Tuple[str, float], str]]
            for category, tags_in_category in raw_result.items():
                if category not in category_config:
                    continue
                toReturn[category] = []
                
                tag_item: Union[Tuple[str, float], str]
                for tag_item in tags_in_category:
                    tagname: str
                    confidence: Optional[float] = None # Initialize confidence
                    current_tag_config: Dict[str, Any]

                    if isinstance(tag_item, tuple) and len(tag_item) == 2:
                        tagname, confidence = tag_item
                        if not isinstance(tagname, str) or not isinstance(confidence, float):
                            logger.warning(f"Malformed tag tuple in {category}: {tag_item}")
                            continue 
                    elif isinstance(tag_item, str):
                        tagname = tag_item
                    else:
                        logger.warning(f"Malformed tag item in {category}: {tag_item}")
                        continue
                    
                    if tagname not in category_config[category]:
                        continue
                    current_tag_config = category_config[category][tagname]
                    
                    tag_threshold: float = float(get_or_default(current_tag_config, 'TagThreshold', 0.5))
                    renamed_tag: str = current_tag_config['RenamedTag']

                    processed_tag: Union[Tuple[str, float], str]
                    if confidence is not None: # Tag came with confidence
                        if not post_processing_config.get("use_category_image_thresholds", False) or confidence >= tag_threshold:
                            processed_tag = (renamed_tag, confidence)
                            toReturn[category].append(processed_tag)
                    else: # Tag came without confidence (just a string)
                        # If no confidence, it usually means it passed a previous threshold or is always included
                        # Here, we might not apply tag_threshold, or assume a default confidence if needed by downstream
                        processed_tag = renamed_tag
                        toReturn[category].append(processed_tag)

            output_target = item.output_names[0] if isinstance(item.output_names, list) else item.output_names
            await itemFuture.set_data(output_target, toReturn)
        except Exception as e:
            logger.error(f"Error in image_result_postprocessor: {e}", exc_info=True)
            itemFuture.set_exception(e)
