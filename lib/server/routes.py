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
            result_data_from_pipeline: Dict[str, Any] = await future
            logger.info(f"Data received directly from AI pipeline (future): {result_data_from_pipeline}")
        except Exception as e:
            logger.error(f"Error during AI processing for video {request.path}: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"AI processing failed: {str(e)}")
        
        json_result_str: Optional[str] = None
        full_ai_output_dict: Optional[Dict[str, Any]] = None

        if isinstance(result_data_from_pipeline, dict) and 'json_result' in result_data_from_pipeline:
            json_input_candidate = result_data_from_pipeline['json_result']
            if isinstance(json_input_candidate, str):
                json_result_str = json_input_candidate
                try:
                    full_ai_output_dict = json.loads(json_result_str)
                except json.JSONDecodeError as e_decode:
                    logger.error(f"Failed to parse json_result_str from pipeline into a dictionary: {e_decode} for video {request.path}")
                    logger.debug(f"json_result_str that failed parsing: {json_result_str[:500]}")
                    raise HTTPException(status_code=500, detail=f"Internal error: AI pipeline returned unparsable JSON data: {e_decode}")
            elif isinstance(json_input_candidate, dict):
                logger.info(f"'json_result' from pipeline is already a dict for video {request.path}. Using it directly and re-serializing for json_result_str.")
                full_ai_output_dict = json_input_candidate
                try:
                    json_result_str = json.dumps(full_ai_output_dict)
                except Exception as e_dump_fallback:
                    logger.error(f"Failed to re-dump full_ai_output_dict (from pipeline dict) to JSON string: {e_dump_fallback} for video {request.path}")
                    raise HTTPException(status_code=500, detail=f"Internal error: Failed to serialize AI results: {e_dump_fallback}")
            else:
                logger.error(f"'json_result' from pipeline is not a string or dict: {type(json_input_candidate)} for video {request.path}")
                raise HTTPException(status_code=500, detail="Internal error: AI pipeline returned malformed json_result.")
        else:
            logger.error(f"Pipeline result does not contain 'json_result' or is not a dict. Got: {result_data_from_pipeline} for video {request.path}")
            raise HTTPException(status_code=500, detail="Internal error: AI pipeline returned unexpected data structure.")

        if full_ai_output_dict is None:
             logger.error(f"full_ai_output_dict is None before building video_tag_info for video {request.path}. This should not happen.")
             raise HTTPException(status_code=500, detail="Internal error: Critical failure in processing AI data.")

        processed_video_duration: float = 0.0
        processed_video_tags: Dict[str, List[str]] = {}
        processed_tag_timespans: Dict[str, Any] = {}
        processed_tag_totals: Dict[str, Any] = {}

        if isinstance(full_ai_output_dict, dict):
            metadata = full_ai_output_dict.get('metadata', {})
            if isinstance(metadata, dict):
                processed_video_duration = metadata.get('duration', 0.0)

            timespans_data = full_ai_output_dict.get('timespans', {})
            processed_tag_timespans = {}

            if isinstance(timespans_data, dict):
                for category, tags_in_category_dict in timespans_data.items():
                    processed_tags_in_category = {}
                    if isinstance(tags_in_category_dict, dict):
                        for tag_name, time_entries_list in tags_in_category_dict.items():
                            processed_time_entries = []
                            if isinstance(time_entries_list, list):
                                for entry in time_entries_list:
                                    if isinstance(entry, dict):
                                        new_entry = {
                                            'start': entry.get('start'),
                                            'end': entry.get('end', entry.get('start')),
                                            'confidence': entry.get('confidence'),
                                            'totalConfidence': entry.get('totalConfidence', entry.get('confidence'))
                                        }
                                        if new_entry['start'] is None or new_entry['confidence'] is None:
                                            logger.warning(f"Skipping entry with missing start/confidence for {tag_name} in video {request.path}: {entry}")
                                            continue
                                        processed_time_entries.append(new_entry)
                                    else:
                                        logger.warning(f"Time entry is not a dict for {tag_name} in video {request.path}: {entry}")
                            processed_tags_in_category[tag_name] = processed_time_entries
                    processed_tag_timespans[category] = processed_tags_in_category
                
                processed_video_tags = {}
                for category, tags_in_category_dict in processed_tag_timespans.items():
                    if isinstance(tags_in_category_dict, dict) and tags_in_category_dict:
                        if list(tags_in_category_dict.keys()): 
                            processed_video_tags[category] = list(tags_in_category_dict.keys())
            else:
                logger.warning(f"'timespans' in parsed AI data (full_ai_output_dict) is not a dict or is missing for video {request.path}.")
        else:
            logger.error(f"full_ai_output_dict is not a dictionary as expected for video {request.path}.")

        video_tag_info_for_client: Dict[str, Any] = {
            "video_duration": processed_video_duration,
            "video_tags": processed_video_tags,
            "tag_totals": processed_tag_totals,
            "tag_timespans": processed_tag_timespans
        }
        logger.debug(f"Constructed video_tag_info_for_client for video {request.path}: {video_tag_info_for_client}")

        final_result_for_client: Dict[str, Any] = {
            "json_result": json_result_str,
            "video_tag_info": video_tag_info_for_client
        }
        logger.trace(f"Final data being wrapped by VideoResult for video {request.path}: {final_result_for_client}")
        
        return VideoResult(result=final_result_for_client)
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