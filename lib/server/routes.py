import asyncio
import json
import logging
from fastapi import HTTPException
from lib.model.postprocessing import tag_models, timeframe_processing
from lib.model.postprocessing.AI_VideoResult import AIVideoResult
from lib.model.preprocessing.input_logic import process_video_preprocess
from lib.server.api_definitions import ImagePathList, OptimizeMarkerSettings, VideoPathList, ImageResult, VideoResult
from lib.server.server_manager import server_manager, app
from lib.model.postprocessing.category_settings import category_config
from typing import List, Any, Dict, Union, Optional

logger: logging.Logger = logging.getLogger("logger")

@app.post("/process_images/")
async def process_images(request: ImagePathList) -> ImageResult:
    try:
        image_paths: List[str] = request.paths
        logger.info(f"Processing {len(image_paths)} images")
        pipeline_name: str = request.pipeline_name or server_manager.default_image_pipeline
        futures: List[asyncio.Future] = [await server_manager.get_request_future([path, request.threshold, request.return_confidence, None], pipeline_name) for path in image_paths]
        results: List[Union[Any, Dict[str, str]]] = await asyncio.gather(*futures, return_exceptions=True)

        for i, result_item in enumerate(results):
            if isinstance(result_item, Exception):
                results[i] = {"error": str(result_item)}

        return_result: ImageResult = ImageResult(result=results)
        logger.debug(f"Returning Image Result: {return_result}")
        return return_result
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process_video/")
async def process_video(request: VideoPathList) -> VideoResult:
    try:
        logger.info(f"Processing video at path: {request.path}")
        pipeline_name: str = request.pipeline_name or server_manager.default_video_pipeline
        
        video_result: Optional[AIVideoResult]
        json_save_needed: bool
        video_result, json_save_needed = AIVideoResult.from_client_json(json_data=request.existing_json_data)

        data: List[Any] = [request.path, request.returnTimestamps, request.frame_interval, request.threshold, request.return_confidence, request.vr_video, None, None]
        if video_result is not None:
            pipeline_to_use: Any = server_manager.pipeline_manager.get_pipeline(pipeline_name)

            ai_work_needed: bool
            skipped_categories: Any
            ai_work_needed, skipped_categories = process_video_preprocess(video_result, request.frame_interval, request.threshold, pipeline_to_use)
            
            if not ai_work_needed:
                json_result: Optional[str] = None
                if json_save_needed:
                    json_result = video_result.to_json()
                
                temp_return_result: Dict[str, Any] = {"json_result": json_result, "video_tag_info": timeframe_processing.compute_video_tag_info(video_result)}

                return VideoResult(result=temp_return_result)
            else:
                data = [request.path, request.returnTimestamps, request.frame_interval, request.threshold, request.return_confidence, request.vr_video, video_result, skipped_categories]

        try:
            future: asyncio.Future = await server_manager.get_request_future(data, pipeline_name)
            result_data: Any = await future
        except Exception as e:
            logger.error(f"Error during AI processing for video {request.path}: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"AI processing failed: {str(e)}")
        
        return_result: VideoResult = VideoResult(result=result_data)
        logger.debug(f"Returning Video Result: {return_result}")
        return return_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/optimize_timeframe_settings/")
async def optimize_timeframe_settings(request: OptimizeMarkerSettings) -> None:
    try:
        video_result: Optional[AIVideoResult]
        _: Any
        video_result, _ = AIVideoResult.from_client_json(json=request.existing_json_data)

        if video_result is None:
            raise HTTPException(status_code=400, detail="Video Result is None")
        else:
            desired_timespan_data: Dict[str, List[str]] = request.desired_timespan_data
            desired_timespan_category_dict: Dict[str, Dict[str, List[tag_models.TimeFrame]]] = {}
            renamedtag_category_dict: Dict[str, str] = {}
            
            current_category_dict: Dict[str, Dict[str, str]]
            for category_key, current_category_dict_val in category_config.items():
                category: str = category_key 
                current_category_dict = current_category_dict_val

                renamed_tag_val: Dict[str, str]
                for tag_key, renamed_tag_val_item in current_category_dict.items():
                    renamed_tag_val = renamed_tag_val_item
                    renamedtag_category_dict[renamed_tag_val["RenamedTag"]] = category
            
            tag_key: str
            time_frames_str: List[str]
            for tag_key, time_frames_str_val in desired_timespan_data.items():
                tag_key = tag_key
                time_frames_str = time_frames_str_val
                category_val: str = renamedtag_category_dict.get(tag_key, "Unknown")
                if category_val not in desired_timespan_category_dict:
                    desired_timespan_category_dict[category_val] = {}
                
                time_frame_str: str
                time_frames_new: List[tag_models.TimeFrame] = [tag_models.TimeFrame(**(json.loads(time_frame_str_item)), totalConfidence=None) for time_frame_str_item in time_frames_str]
                desired_timespan_category_dict[category_val][tag_key] = time_frames_new
            timeframe_processing.determine_optimal_timespan_settings(video_result, desired_timespan_data=desired_timespan_category_dict)
        return
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing timeframe settings: {e}")
        logger.debug("Stack trace:", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))