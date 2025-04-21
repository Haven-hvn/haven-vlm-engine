
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
from lib.model.vlm_model import VLMModel

logger = logging.getLogger("logger")

# Global VLM model instance
_vlm_model = None

def get_vlm_model(tag_list):
    """Get or create the VLM model instance"""
    global _vlm_model
    if _vlm_model is None:
        logger.info(f"Initializing VLM model with tags: {tag_list}")
        _vlm_model = VLMModel(
            tag_list=tag_list,
            use_quantization=True,
            device=None,  # Will use CUDA if available
        )
    return _vlm_model

async def vlm_frame_analyzer(data):
    """
    Process a frame using the VLM model.
    Returns a dictionary of {tag: confidence} for each tag in the tag list.
    """
    for item in data:
        try:
            item_future = item.item_future
            frame_tensor = item_future[item.input_names[0]]
            tag_list = item_future[item.input_names[1]]
            
            # Convert tensor to PIL Image
            # Assuming the frame is already preprocessed and normalized
            # We need to denormalize and convert to PIL Image
            frame_pil = Image.fromarray(frame_tensor.cpu().numpy())
            
            # Get or initialize the VLM model
            vlm = get_vlm_model(tag_list)
            
            # Analyze the frame
            scores = vlm.analyze_frame(frame_pil)
            
            # Set the result
            await item_future.set_data(item.output_names[0], scores)
        except Exception as e:
            logger.error(f"Error in VLM frame analyzer: {e}")
            logger.debug("Stack trace:", exc_info=True)
            item_future.set_exception(e)

async def result_coalescer(data):
    for item in data:
        itemFuture = item.item_future
        result = {}
        for input_name in item.input_names:
            ai_result = itemFuture[input_name]
            if not isinstance(ai_result, Skip):
                result[input_name] = itemFuture[input_name]
        await itemFuture.set_data(item.output_names[0], result)
        
async def result_finisher(data):
    for item in data:
        itemFuture = item.item_future
        future_results = itemFuture[item.input_names[0]]
        itemFuture.close_future(future_results)

async def batch_awaiter(data):
    for item in data:
        itemFuture = item.item_future
        futures = itemFuture[item.input_names[0]]
        results = await asyncio.gather(*futures, return_exceptions=True)
        await itemFuture.set_data(item.output_names[0], results)

async def video_result_postprocessor(data):
    for item in data:
        itemFuture = item.item_future
        duration = get_video_duration_decord(itemFuture[item.input_names[1]])
        result = {"frames": itemFuture[item.input_names[0]], "video_duration": duration, "frame_interval": float(itemFuture[item.input_names[2]]), "threshold": float(itemFuture[item.input_names[3]]), "ai_models_info": itemFuture['pipeline'].get_ai_models_info()}
        del itemFuture.data["pipeline"]

        videoResult = itemFuture[item.input_names[4]]
        if videoResult is not None:
            videoResult.add_server_result(result)
        else:
            videoResult = AIVideoResult.from_server_result(result)

        toReturn = {"json_result": videoResult.to_json(), "video_tag_info": timeframe_processing.compute_video_tag_info(videoResult)}
        
        await itemFuture.set_data(item.output_names[0], toReturn)

async def image_result_postprocessor(data):
    toReturn = {}
    for item in data:
        itemFuture = item.item_future
        result = itemFuture[item.input_names[0]]
        for category, tags in result.items():
            if category not in category_config:
                continue
            toReturn[category] = []
            for tag in tags:
                if isinstance(tag, tuple):
                    tagname, confidence = tag
                    if tagname not in category_config[category]:
                        continue
                    
                    tag_threshold = float(get_or_default(category_config[category][tagname], 'TagThreshold', 0.5))
                    renamed_tag = category_config[category][tagname]['RenamedTag']

                    if not post_processing_config["use_category_image_thresholds"]:
                        toReturn[category].append((renamed_tag, confidence))
                    elif confidence >= tag_threshold:
                        toReturn[category].append((renamed_tag, confidence))
                else:
                    if tag not in category_config[category]:
                        continue
                    renamed_tag = category_config[category][tag]['RenamedTag']
                    toReturn[category].append(renamed_tag)


        await itemFuture.set_data(item.output_names[0], toReturn)
