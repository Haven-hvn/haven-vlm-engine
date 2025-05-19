from copy import deepcopy
import logging
import math
from lib.model.postprocessing.category_settings import category_config
import lib.model.postprocessing.tag_models as tag_models
from lib.model.postprocessing.post_processing_settings import get_or_default, post_processing_config
from lib.model.postprocessing.AI_VideoResult import AIVideoResult, TagTimeFrame
from typing import Dict, List, Optional, Set, Any, Union, Tuple, Callable
import itertools

logger: logging.Logger = logging.getLogger("logger")

def compute_video_tag_info(video_result: AIVideoResult) -> tag_models.VideoTagInfo:
    video_timespans: Dict[str, Dict[str, List[tag_models.TimeFrame]]] = compute_video_timespans(video_result)
    video_tags_data: Tuple[Dict[str, Set[str]], Dict[str, Dict[str, float]]] = compute_video_tags(video_result)
    video_tags: Dict[str, Set[str]] = video_tags_data[0]
    tag_totals: Dict[str, Dict[str, float]] = video_tags_data[1]
    return tag_models.VideoTagInfo(
        video_duration=video_result.metadata.duration, 
        video_tags=video_tags, 
        tag_totals=tag_totals, 
        tag_timespans=video_timespans
    )

def compute_video_timespans_OG(video_result: AIVideoResult) -> Dict[str, Dict[str, List[tag_models.TimeFrame]]]:
    video_duration: float = video_result.metadata.duration
    toReturn: Dict[str, Dict[str, List[tag_models.TimeFrame]]] = {}
    
    category: str
    tag_to_raw_timespans_map: Dict[str, List[TagTimeFrame]]
    for category, tag_to_raw_timespans_map in video_result.timespans.items():
        if category not in category_config:
            logger.debug(f"Category {category} not found in category settings")
            continue
        
        frame_interval: float = 0.5 # Default, or retrieve more robustly
        if hasattr(video_result.metadata, 'models') and category in video_result.metadata.models and hasattr(video_result.metadata.models[category], 'frame_interval'):
            frame_interval = float(video_result.metadata.models[category].frame_interval)
        else:
            logger.warning(f"Frame interval not found for category {category} in video_result metadata. Using default.")

        toReturn[category] = {}
        tag: str
        raw_timespans_list: List[TagTimeFrame]
        for tag, raw_timespans_list in tag_to_raw_timespans_map.items():
            if tag not in category_config[category]:
                continue
            
            tag_min_duration: float = format_duration_or_percent(get_or_default(category_config[category][tag], 'MinMarkerDuration', "12s"), video_duration)
            if tag_min_duration <= 0:
                continue
            tag_threshold: float = float(get_or_default(category_config[category][tag], 'TagThreshold', 0.5))
            tag_max_gap: float = format_duration_or_percent(get_or_default(category_config[category][tag], 'MaxGap', "6s"), video_duration)
            renamed_tag: str = category_config[category][tag]['RenamedTag']

            processed_timeframes: List[TagTimeFrame] = []
            raw_timespan_obj: TagTimeFrame
            for raw_timespan_obj in raw_timespans_list:
                if raw_timespan_obj.confidence is None or raw_timespan_obj.confidence < tag_threshold:
                    continue
                
                if not processed_timeframes:
                    processed_timeframes.append(deepcopy(raw_timespan_obj))
                    continue
                else:
                    previous_tf_obj: TagTimeFrame = processed_timeframes[-1]
                    if not hasattr(raw_timespan_obj, 'start') or not hasattr(previous_tf_obj, 'start'): continue
                    
                    current_previous_end: float = previous_tf_obj.end if hasattr(previous_tf_obj, 'end') and previous_tf_obj.end is not None else previous_tf_obj.start
                    current_raw_start: float = raw_timespan_obj.start
                    current_raw_end: float = raw_timespan_obj.end if hasattr(raw_timespan_obj, 'end') and raw_timespan_obj.end is not None else raw_timespan_obj.start

                    if current_raw_start - current_previous_end - frame_interval <= tag_max_gap:
                        previous_tf_obj.end = current_raw_end
                    else:
                        processed_timeframes.append(deepcopy(raw_timespan_obj))
            
            final_tag_timeframes: List[tag_models.TimeFrame] = [
                tag_models.TimeFrame(start=tf.start, end=(tf.end if hasattr(tf, 'end') and tf.end is not None else tf.start), totalConfidence=None) 
                for tf in processed_timeframes 
                if hasattr(tf, 'start') and 
                   ( (hasattr(tf, 'end') and tf.end is not None and (tf.end - tf.start >= tag_min_duration)) or 
                     ( (not hasattr(tf, 'end') or tf.end is None) and tag_min_duration <= frame_interval)
                   )
            ]
            if final_tag_timeframes:
                toReturn[category][renamed_tag] = final_tag_timeframes
    return toReturn

def compute_video_timespans_clustering(
    video_result: AIVideoResult, 
    density_weight: float, 
    gap_factor: float, 
    average_factor: float, 
    min_gap: float
) -> Dict[str, Dict[str, List[tag_models.TimeFrame]]]:
    video_duration: float = video_result.metadata.duration
    toReturn: Dict[str, Dict[str, List[tag_models.TimeFrame]]] = {}

    category: str
    tag_to_raw_timespans_map: Dict[str, List[TagTimeFrame]]
    for category, tag_to_raw_timespans_map in video_result.timespans.items():
        if category not in category_config:
            logger.debug(f"Category {category} not found in category config")
            continue
        
        frame_interval: float = 0.5 # Default
        if hasattr(video_result.metadata, 'models') and category in video_result.metadata.models and hasattr(video_result.metadata.models[category], 'frame_interval'):
            frame_interval = float(video_result.metadata.models[category].frame_interval)
        else:
            logger.warning(f"Frame interval for category {category} not found, using default.")

        toReturn[category] = {}
        
        tag: str
        raw_timespans_list: List[TagTimeFrame]
        for tag, raw_timespans_list in tag_to_raw_timespans_map.items():
            if tag not in category_config[category]:
                continue
            
            tag_threshold: float = float(get_or_default(category_config[category][tag], 'TagThreshold', 0.5))
            renamed_tag: str = category_config[category][tag]['RenamedTag']
            tag_min_duration: float = format_duration_or_percent(
                get_or_default(category_config[category][tag], 'MinMarkerDuration', "12s"),
                video_duration
            )
            if tag_min_duration <= 0:
                continue
            
            initial_buckets: List[tag_models.TimeFrame] = []
            current_bucket: Optional[tag_models.TimeFrame] = None
            
            raw_timespan_obj: TagTimeFrame
            for i, raw_timespan_obj in enumerate(raw_timespans_list):
                if raw_timespan_obj.confidence is None:
                    continue
                confidence: float = raw_timespan_obj.confidence
                if confidence < tag_threshold:
                    continue

                start_time: float = raw_timespan_obj.start
                end_time: float = raw_timespan_obj.end if hasattr(raw_timespan_obj, 'end') and raw_timespan_obj.end is not None else start_time
                
                duration_val: float = (end_time - start_time) + frame_interval
                if current_bucket is None:
                    current_bucket = tag_models.TimeFrame(start=start_time, end=end_time, totalConfidence=confidence * duration_val)
                else:
                    if start_time - current_bucket.end == frame_interval:
                        current_bucket.merge(math.floor(start_time), math.floor(end_time), confidence, frame_interval)
                    else:
                        initial_buckets.append(current_bucket)
                        current_bucket = tag_models.TimeFrame(start=start_time, end=end_time, totalConfidence=confidence * duration_val)
            
            if current_bucket is not None:
                initial_buckets.append(current_bucket)
            
            def should_merge_buckets(b1: tag_models.TimeFrame, b2: tag_models.TimeFrame, fi: float, dw: float, gf: float, af: float, mg: float) -> bool:
                gap: float = b2.start - b1.end - fi
                dur_b1: float = b1.get_duration(fi)
                dur_b2: float = b2.get_duration(fi)
                den_b1: float = b1.get_density(fi)
                den_b2: float = b2.get_density(fi)

                wd_b1: float = dur_b1 * (1 + dw * den_b1)
                wd_b2: float = dur_b2 * (1 + dw * den_b2)
                wd_diff: float = abs(wd_b1 - wd_b2)
                return (gap <= mg + (min(wd_b1, wd_b2) + wd_diff * af) * gf)

            merged_buckets: List[tag_models.TimeFrame] = initial_buckets
            merging_occurred: bool = True
            max_iterations: int = 10
            iterations_count: int = 0
            while merging_occurred and iterations_count < max_iterations:
                merging_occurred = False
                new_buckets_list: List[tag_models.TimeFrame] = []
                idx: int = 0
                iterations_count += 1

                while idx < len(merged_buckets):
                    if idx < len(merged_buckets) - 1 and should_merge_buckets(merged_buckets[idx], merged_buckets[idx + 1], frame_interval, density_weight, gap_factor, average_factor, min_gap):
                        b1_to_merge: tag_models.TimeFrame = merged_buckets[idx]
                        b2_to_merge: tag_models.TimeFrame = merged_buckets[idx + 1]
                        tc1: float = b1_to_merge.totalConfidence if b1_to_merge.totalConfidence is not None else 0.0
                        tc2: float = b2_to_merge.totalConfidence if b2_to_merge.totalConfidence is not None else 0.0
                        new_merged_bucket: tag_models.TimeFrame = tag_models.TimeFrame(
                            start=b1_to_merge.start,
                            end=b2_to_merge.end,
                            totalConfidence=tc1 + tc2
                        )
                        new_buckets_list.append(new_merged_bucket)
                        idx += 2
                        merging_occurred = True
                    else:
                        new_buckets_list.append(merged_buckets[idx])
                        idx += 1
                merged_buckets = new_buckets_list

            final_buckets_list: List[tag_models.TimeFrame] = [
                b for b in merged_buckets if b.get_duration(frame_interval) >= tag_min_duration
            ]

            if final_buckets_list:
                toReturn[category][renamed_tag] = final_buckets_list
    return toReturn

def compute_video_timespans_proportional_merge(video_result: AIVideoResult, prop: float = 0.5) -> Dict[str, Dict[str, List[tag_models.TimeFrame]]]:
    video_duration: float = video_result.metadata.duration
    toReturn: Dict[str, Dict[str, List[tag_models.TimeFrame]]] = {}
    
    category: str
    tag_to_raw_timespans_map: Dict[str, List[TagTimeFrame]]
    for category, tag_to_raw_timespans_map in video_result.timespans.items():
        if category not in category_config:
            logger.debug(f"Category {category} not found in category settings")
            continue
        
        frame_interval: float = 0.5 # Default
        if hasattr(video_result.metadata, 'models') and category in video_result.metadata.models and hasattr(video_result.metadata.models[category], 'frame_interval'):
            frame_interval = float(video_result.metadata.models[category].frame_interval)
        else:
            logger.warning(f"Frame interval for category {category} not found, using default.")

        toReturn[category] = {}
        tag: str
        raw_timespans_list: List[TagTimeFrame]
        for tag, raw_timespans_list in tag_to_raw_timespans_map.items():
            if tag not in category_config[category]:
                continue
            
            tag_threshold: float = float(get_or_default(category_config[category][tag], 'TagThreshold', 0.5))
            tag_max_gap_config: float = format_duration_or_percent(get_or_default(category_config[category][tag], 'MaxGap', "6s"), video_duration)
            tag_min_duration: float = format_duration_or_percent(get_or_default(category_config[category][tag], 'MinMarkerDuration', "12s"), video_duration)
            renamed_tag: str = category_config[category][tag]['RenamedTag']

            if tag_min_duration <= 0:
                continue

            processed_tag_timeframes: List[TagTimeFrame] = []
            raw_timespan_obj: TagTimeFrame
            for raw_timespan_obj in raw_timespans_list:
                if raw_timespan_obj.confidence is None or raw_timespan_obj.confidence < tag_threshold:
                    continue
                if not hasattr(raw_timespan_obj, 'start'): continue
                
                if not processed_tag_timeframes:
                    processed_tag_timeframes.append(deepcopy(raw_timespan_obj))
                    continue
                
                previous_tf_obj: TagTimeFrame = processed_tag_timeframes[-1]
                if not hasattr(previous_tf_obj, 'start'): continue

                current_raw_start: float = raw_timespan_obj.start
                current_raw_end: float = raw_timespan_obj.end if raw_timespan_obj.end is not None else current_raw_start
                prev_start: float = previous_tf_obj.start
                prev_end: float = previous_tf_obj.end if previous_tf_obj.end is not None else prev_start

                gap_with_prev: float = (current_raw_start - prev_end) - frame_interval
                last_segment_duration: float = (prev_end - prev_start) + frame_interval

                if gap_with_prev <= tag_max_gap_config or (last_segment_duration > 0 and gap_with_prev <= prop * last_segment_duration):
                    previous_tf_obj.end = current_raw_end
                else:
                    processed_tag_timeframes.append(deepcopy(raw_timespan_obj))

            final_tag_timeframes_prop: List[tag_models.TimeFrame] = [
                tag_models.TimeFrame(start=tf.start, end=(tf.end if tf.end is not None else tf.start), totalConfidence=None)
                for tf in processed_tag_timeframes 
                if ( (tf.end is not None and ((tf.end - tf.start) + frame_interval >= tag_min_duration)) or
                     ( (tf.end is None) and (frame_interval >= tag_min_duration) )
                   )
            ]
            if final_tag_timeframes_prop:
                toReturn[category][renamed_tag] = final_tag_timeframes_prop
    return toReturn

ClusteringParams = Dict[str, float]
ProportionalMergeParams = Dict[str, float]
VideoTimespanCallable = Callable[..., Dict[str, Dict[str, List[tag_models.TimeFrame]]]]

active_timespan_method: str = "Clustering"
timespan_methods: Dict[str, VideoTimespanCallable] = {
    "OG": compute_video_timespans_OG,
    "Clustering": compute_video_timespans_clustering,
    "Proportional_Merge": compute_video_timespans_proportional_merge
}

timespan_configuration_defaults: Dict[str, Dict[str, Any]] = {
    "OG": {},
    "Clustering": {
        "density_weight": post_processing_config.get('timespan_clustering_density_weight', 0.2),
        "gap_factor": post_processing_config.get('timespan_clustering_gap_factor', 0.75),
        "average_factor": post_processing_config.get('timespan_clustering_average_factor', 0.5),
        "min_gap": post_processing_config.get('timespan_clustering_min_gap', 1.0)
    },
    "Proportional_Merge": {
        "prop": post_processing_config.get('timespan_proportional_merge_prop', 0.5)
    }
}

def load_post_processing_settings(config: Optional[Dict[str, Any]] = None) -> None:
    global active_timespan_method, timespan_configuration_defaults
    if config is None:
        config = post_processing_config

    active_timespan_method = str(config.get('active_timespan_method', "Clustering"))
    
    clustering_config: Dict[str, Any] = timespan_configuration_defaults.get("Clustering", {})
    clustering_config["density_weight"] = float(config.get('timespan_clustering_density_weight', clustering_config.get("density_weight", 0.2)))
    clustering_config["gap_factor"] = float(config.get('timespan_clustering_gap_factor', clustering_config.get("gap_factor", 0.75)))
    clustering_config["average_factor"] = float(config.get('timespan_clustering_average_factor', clustering_config.get("average_factor", 0.5)))
    clustering_config["min_gap"] = float(config.get('timespan_clustering_min_gap', clustering_config.get("min_gap", 1.0)))
    timespan_configuration_defaults["Clustering"] = clustering_config
    
    prop_merge_config: Dict[str, Any] = timespan_configuration_defaults.get("Proportional_Merge", {})
    prop_merge_config["prop"] = float(config.get('timespan_proportional_merge_prop', prop_merge_config.get("prop", 0.5)))
    timespan_configuration_defaults["Proportional_Merge"] = prop_merge_config

load_post_processing_settings()

def compute_video_timespans(video_result: AIVideoResult) -> Dict[str, Dict[str, List[tag_models.TimeFrame]]]:
    method_func: Optional[VideoTimespanCallable] = timespan_methods.get(active_timespan_method)
    
    if not method_func:
        logger.error(f"Timespan method '{active_timespan_method}' not found. Falling back to OG.")
        return compute_video_timespans_OG(video_result)
    
    current_params: Dict[str, Any] = timespan_configuration_defaults.get(active_timespan_method, {})
    
    try:
        if active_timespan_method == "OG":
            return method_func(video_result)
        else:
            return method_func(video_result, **current_params)
    except Exception as e:
        logger.error(f"Error calling timespan method {active_timespan_method} with params {current_params}: {e}", exc_info=True)
        logger.warning("Falling back to OG timespan computation.")
        return compute_video_timespans_OG(video_result)

def determine_optimal_timespan_settings(
    video_result: AIVideoResult, 
    desired_timespan_data: Dict[str, Dict[str, List[tag_models.TimeFrame]]],
    timespan_configuration_sweep: Dict[str, List[List[Any]]]
) -> None:
    def total_tag_duration(spans: List[tag_models.TimeFrame]) -> float:
        return sum((m.end - m.start) for m in spans if m.end is not None and m.start is not None)

    def intersection_coverage(A: List[tag_models.TimeFrame], B: List[tag_models.TimeFrame]) -> float:
        i: int = 0
        j: int = 0
        overlap: float = 0.0
        while i < len(A) and j < len(B):
            startA: float = A[i].start
            endA: float = A[i].end if A[i].end is not None else A[i].start
            startB: float = B[j].start
            endB: float = B[j].end if B[j].end is not None else B[j].start

            inter_start: float = max(startA, startB)
            inter_end: float = min(endA, endB)
            if inter_start < inter_end:
                overlap += (inter_end - inter_start)

            if endA < endB:
                i += 1
            else:
                j += 1
        return overlap

    def measure_loss(
        actual_timespans: Dict[str, Dict[str, List[tag_models.TimeFrame]]], 
        desired_timespans: Dict[str, Dict[str, List[tag_models.TimeFrame]]]
    ) -> float:
        shared_categories: Set[str] = set(actual_timespans.keys()).union(desired_timespans.keys())
        total_mismatch: float = 0.0

        cat: str
        for cat in shared_categories:
            actual_tag_timeframes_map: Dict[str, List[tag_models.TimeFrame]] = actual_timespans.get(cat, {})
            desired_tags_map: Dict[str, List[tag_models.TimeFrame]] = desired_timespans.get(cat, {})

            all_tags: Set[str] = set(desired_tags_map.keys())
            tag: str
            for tag in all_tags:
                actual_tag_timeframes_list: List[tag_models.TimeFrame] = actual_tag_timeframes_map.get(tag, [])
                desired_tag_timeframes_list: List[tag_models.TimeFrame] = desired_tags_map.get(tag, [])

                actual_total_tag_time: float = total_tag_duration(actual_tag_timeframes_list)
                desired_total_tag_time: float = total_tag_duration(desired_tag_timeframes_list)

                if actual_total_tag_time == 0.0 and desired_total_tag_time == 0.0:
                    mismatch: float = 0.0
                else:
                    inter: float = intersection_coverage(actual_tag_timeframes_list, desired_tag_timeframes_list)
                    mismatch = (actual_total_tag_time + desired_total_tag_time) - 2.0 * inter
                total_mismatch += mismatch
        return total_mismatch
    
    def mismatch_count(
        actual_timespans: Dict[str, Dict[str, List[tag_models.TimeFrame]]], 
        desired_timespans: Dict[str, Dict[str, List[tag_models.TimeFrame]]]
    ) -> int:
        actual_spans: int = sum(len(spans) for cat_dict in actual_timespans.values() for spans in cat_dict.values())
        desired_spans: int = sum(len(spans) for cat_dict in desired_timespans.values() for spans in cat_dict.values())
        return abs(actual_spans - desired_spans)

    used_method: Optional[VideoTimespanCallable] = timespan_methods.get(active_timespan_method)
    if not used_method:
        logger.error(f"Active method {active_timespan_method} not found in timespan_methods for optimization.")
        return

    sweep_lists: List[List[Any]] = timespan_configuration_sweep.get(active_timespan_method, [])

    if not sweep_lists:
        logger.debug("No sweep lists defined for the active method.")
        return

    param_combos: List[Tuple[Any, ...]] = list(set(itertools.product(*sweep_lists)))
    results: List[Tuple[float, Tuple[Any, ...], int, float, int]] = []

    combo: Tuple[Any, ...]
    for combo in param_combos:
        try:
            timespans_output: Dict[str, Dict[str, List[tag_models.TimeFrame]]] = used_method(video_result, *combo)
            combo_loss: float = measure_loss(timespans_output, desired_timespan_data)
            
            total_spans_val: int = 0
            cat_dict_val: Dict[str, List[tag_models.TimeFrame]]
            spans_val: List[tag_models.TimeFrame]
            for cat_dict_val in timespans_output.values():
                for spans_val in cat_dict_val.values():
                    total_spans_val += len(spans_val)
            
            mismatch_total_val: int = mismatch_count(timespans_output, desired_timespan_data)
            results.append((combo_loss + 2 * mismatch_total_val, combo, total_spans_val, combo_loss, mismatch_total_val))
        except Exception as e:
            logger.debug(f"Error with combo {combo} for method {active_timespan_method}: {e}")
            logger.debug("Stack trace:", exc_info=True)

    results.sort(key=lambda x: x[0])

    logger.info("Top 10 parameter combos (lowest mismatch):")
    i: int
    loss_val: float
    combo: Tuple[Any, ...]
    total_spans_log: int
    seconds_diff: float
    mismatch_total_log: int
    for i, (loss_val, combo, total_spans_log, seconds_diff, mismatch_total_log) in enumerate(results[:10]):
        logger.info(f"#{i+1} Loss={loss_val:.4f} | Timespans={total_spans_log} | Params={combo} | SecondsDiff={seconds_diff:.2f} | MismatchCount={mismatch_total_log}")

def compute_video_tags(video_result: AIVideoResult) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
    return compute_video_tags_OG(video_result)

def compute_video_tags_OG(video_result: AIVideoResult) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
    video_tags: Dict[str, Set[str]] = {}
    tag_totals: Dict[str, Dict[str, float]] = {}
    video_duration: float = video_result.metadata.duration
    
    category: str
    tag_raw_timespans_map: Dict[str, List[TagTimeFrame]]
    for category, tag_raw_timespans_map in video_result.timespans.items():
        if category not in category_config:
            logger.debug(f"Category {category} not found in category settings")
            continue
        video_tags[category] = set()
        tag_totals[category] = {}
        
        frame_interval: float = 0.5 # Default
        if hasattr(video_result.metadata, 'models') and category in video_result.metadata.models:
            frame_interval = float(video_result.metadata.models[category].frame_interval)
        else:
            logger.warning(f"Frame interval for category {category} not found in metadata, using default {frame_interval}")

        tag: str
        raw_timespans_list: List[TagTimeFrame]
        for tag, raw_timespans_list in tag_raw_timespans_map.items():
            if tag not in category_config[category]:
                logger.debug(f"Tag {tag} not found in category settings for category {category}")
                continue
            
            required_duration_val: Any = get_or_default(category_config[category][tag], 'RequiredDuration', "20s")
            required_duration: float = format_duration_or_percent(required_duration_val, video_duration)
            
            tag_threshold_val: Any = get_or_default(category_config[category][tag], 'TagThreshold', 0.5)
            tag_threshold: float = float(tag_threshold_val)
            
            totalDuration: float = 0.0
            raw_timespan: TagTimeFrame
            for raw_timespan in raw_timespans_list:
                if raw_timespan.confidence is not None and raw_timespan.confidence < tag_threshold:
                    continue
                if raw_timespan.end is None:
                    totalDuration += frame_interval
                else:
                    totalDuration += (raw_timespan.end - raw_timespan.start) + frame_interval
            
            renamed_tag: str = category_config[category][tag]['RenamedTag']
            tag_totals[category][renamed_tag] = totalDuration
            if required_duration > 0 and totalDuration >= required_duration:
                video_tags[category].add(renamed_tag)
    return video_tags, tag_totals

def format_duration_or_percent(value: Union[str, float, int], video_duration: float) -> float:
    try:
        if isinstance(value, float):
            return value
        elif isinstance(value, str):
            if value.endswith('%'):
                return (float(value[:-1]) / 100.0) * video_duration
            elif value.endswith('s'):
                return float(value[:-1])
            else:
                return float(value)
        elif isinstance(value, int):
            return float(value)
        logger.warning(f"Unsupported type for format_duration_or_percent: {type(value)}, value: {value}")
        return 0.0
    except ValueError as e:
        logger.error(f"Error in format_duration_or_percent converting value '{value}': {e}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error in format_duration_or_percent: {e} for value '{value}'")
        logger.debug("Stack trace:", exc_info=True)
        return 0.0