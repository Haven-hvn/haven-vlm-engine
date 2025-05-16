import decord
import torch
from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import read_image, VideoReader
import torchvision
from typing import List, Tuple, Dict, Any, Optional, Union, Iterator
import logging
from PIL import Image as PILImage

decord.bridge.set_bridge('torch')

def custom_round(number: float) -> int:
    if number - int(number) >= 0.5:
        return int(number) + 1
    else:
        return int(number)

def get_normalization_config(config_index: Union[int, Dict[str, List[float]]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Default to configuration 1 if an invalid index is somehow passed and it's not a direct dict
    if isinstance(config_index, int) and config_index not in [1, 2, 3]:
        config_index = 1

    if isinstance(config_index, dict):
        mean_values = config_index.get("mean", [0.485, 0.456, 0.406]) # Default if not in dict
        std_values = config_index.get("std", [0.229, 0.224, 0.225]) # Default if not in dict
        mean = torch.tensor(mean_values, device=device)
        std = torch.tensor(std_values, device=device)
    elif config_index == 1: # Default (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        std = torch.tensor([0.229, 0.224, 0.225], device=device)
    elif config_index == 2: # Values for some other common models, e.g., CLIP
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
    elif config_index == 3: # No normalization / passthrough (mean 0, std 1)
        mean = torch.tensor([0.0, 0.0, 0.0], device=device)
        std = torch.tensor([1.0, 1.0, 1.0], device=device)
    else: # Should not happen due to check above, but as a fallback
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        std = torch.tensor([0.229, 0.224, 0.225], device=device)
    return mean, std

def get_video_duration_torchvision(video_path: str) -> float:
    video: VideoReader = torchvision.io.VideoReader(video_path, "video")
    metadata: Dict[str, Any] = video.get_metadata()
    duration: float = 0.0
    if metadata and 'video' in metadata and metadata['video']['duration'] and metadata['video']['duration'][0]:
        duration = float(metadata['video']['duration'][0])
    return duration

def get_video_duration_decord(video_path: str) -> float:
    try:
        vr: decord.VideoReader = decord.VideoReader(video_path, ctx=decord.cpu(0))
        num_frames: int = len(vr)
        frame_rate: float = vr.get_avg_fps()
        if frame_rate == 0: return 0.0
        duration: float = num_frames / frame_rate
        del vr
        return duration
    except RuntimeError as e:
        logging.getLogger("logger").error(f"Decord could not read video {video_path}: {e}")
        return 0.0

def get_frame_transforms(use_half_precision: bool, mean: torch.Tensor, std: torch.Tensor, vr_video: bool, img_size: Union[int, Tuple[int, int]]) -> transforms.Compose:
    normalize_transform = transforms.Normalize(mean=mean, std=std)
    
    transform_list = []
    # VR videos might need specific handling for ToTensor or initial conversions if not standard HWC
    if vr_video:
        # Specific VR frame adjustments (e.g., cropping, view selection)
        # are handled by the vr_permute() function prior to these transforms.
        pass

    # Handle both int (square size) and Tuple[int, int] (height, width) for img_size
    if isinstance(img_size, int):
        target_size = (img_size, img_size)
    elif isinstance(img_size, tuple) and len(img_size) == 2:
        target_size = img_size
    else: # Fallback or error for invalid img_size
        logging.getLogger("logger").warning(f"Invalid img_size {img_size}, defaulting to 224x224 for transforms.")
        target_size = (224, 224)

    transform_list.extend([
        transforms.ToDtype(torch.half if use_half_precision else torch.float, scale=True), # Converts to right dtype and scales (e.g. uint8 to float [0,1])
        transforms.Resize(target_size, antialias=True), # type: ignore
        normalize_transform
    ])
    
    return transforms.Compose(transform_list)

def vr_permute(frame: torch.Tensor) -> torch.Tensor:
    # Restored VR permutation/cropping logic.
    # Assumes HWC input tensor.
    logger = logging.getLogger("logger")
    if frame.ndim != 3 or frame.shape[2] != 3: 
        logger.warning(f"vr_permute received unexpected frame shape: {frame.shape}. Expecting HWC format.")
        return frame

    height: int = frame.shape[0]
    width: int = frame.shape[1]
    
    if height == 0 or width == 0:
        logger.warning(f"vr_permute received frame with zero height or width: {frame.shape}")
        return frame

    aspect_ratio: float = width / height

    if aspect_ratio > 1.5: # Likely side-by-side format, take left half
        return frame[:, :width//2, :] 
    else: # Other formats (e.g., potentially over-under, or specific crop)
        # This was the original logic: takes top half of height, and middle half of width.
        return frame[:height//2, width//4:(width//4 + width//2), :] # Corrected slicing for middle half

def preprocess_image(image_path_or_pil: Union[str, PILImage.Image, torch.Tensor], img_size: Union[int, Tuple[int,int]] = 512, use_half_precision: bool = True, device_str: Optional[str] = None, norm_config_idx: int = 1) -> torch.Tensor:
    actual_device: torch.device = torch.device(device_str) if device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = get_normalization_config(norm_config_idx, actual_device)
    target_dtype: torch.dtype = torch.float16 if use_half_precision else torch.float32

    img_tensor: torch.Tensor
    if isinstance(image_path_or_pil, str):
        img_tensor = read_image(image_path_or_pil)
    elif isinstance(image_path_or_pil, PILImage.Image):
        img_tensor = transforms.functional.pil_to_tensor(image_path_or_pil)
    elif isinstance(image_path_or_pil, torch.Tensor):
        img_tensor = image_path_or_pil
    else:
        raise TypeError(f"Unsupported input type for preprocess_image: {type(image_path_or_pil)}")
    
    img_tensor = img_tensor.to(actual_device)
    if img_tensor.ndim == 2: 
        img_tensor = img_tensor.unsqueeze(0) 
    if img_tensor.shape[0] == 1: 
        img_tensor = img_tensor.repeat(3, 1, 1) 
    elif img_tensor.shape[0] == 4: 
        img_tensor = img_tensor[:3, :, :]
    
    transform_list_img: List[Any] = []
    
    current_img_size_for_resize: Union[int, Tuple[int,int]]
    if isinstance(img_size, int):
        current_img_size_for_resize = (img_size, img_size)
    else:
        current_img_size_for_resize = img_size

    transform_list_img.append(transforms.Resize(current_img_size_for_resize, interpolation=InterpolationMode.BICUBIC, antialias=True))
    transform_list_img.append(transforms.ToDtype(target_dtype, scale=True))
    transform_list_img.append(transforms.Normalize(mean=mean, std=std))
    
    image_transforms_comp: transforms.Compose = transforms.Compose(transform_list_img)
    return image_transforms_comp(img_tensor)
    
def preprocess_video(video_path: str, frame_interval_sec: float = 0.5, img_size: Union[int, Tuple[int,int]] = 512, use_half_precision: bool = True, device_str: Optional[str] = None, use_timestamps: bool = False, vr_video: bool = False, norm_config_idx: int = 1, process_for_vlm: bool = False) -> Iterator[Tuple[Union[int, float], torch.Tensor]]:
    actual_device: torch.device = torch.device(device_str) if device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = logging.getLogger("logger") # Use a logger instance

    try:
        # decord.VideoReader by default gives frames that can be converted via .asnumpy()
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    except RuntimeError as e:
        logger.error(f"Decord failed to open video {video_path}: {e}")
        return
        
    fps: float = vr.get_avg_fps()
    if fps == 0:
        logger.warning(f"Video {video_path} has FPS of 0. Cannot process.")
        if 'vr' in locals(): del vr # Ensure vr is deleted if initialized
        return

    # Assuming custom_round is defined elsewhere in the file or imported
    frames_to_skip: int = custom_round(fps * frame_interval_sec) 
    if frames_to_skip < 1: frames_to_skip = 1

    if process_for_vlm:
        for i in range(0, len(vr), frames_to_skip):
            try:
                frame = vr[i].to(actual_device) # HWC, RGB, (likely) uint8 tensor on device
            except RuntimeError as e_read_frame:
                logger.warning(f"Could not read frame {i} from {video_path}: {e_read_frame}")
                continue
            
            if not torch.is_floating_point(frame): # If uint8
                frame = frame.float() # Convert to float, but keep original range (e.g., 0-255)
            # else: frame is already float, assume its range is what VLM expects

            # Crop black bars from left and right sides
            frame = crop_black_bars_lr(frame)
            
            if use_half_precision:
                frame = frame.half()
            
            if vr_video: 
                # Assuming vr_permute is defined elsewhere and handles HWC input, returning HWC
                frame = vr_permute(frame) 
            
            # VLM receives HWC, RGB, float/half, [0,1] scaled tensor
            transformed_frame = frame
            
            frame_identifier: Union[int, float] = i / fps if use_timestamps else i
            yield (frame_identifier, transformed_frame)
    else: # Standard processing path
        # Assuming get_normalization_config and get_frame_transforms are defined elsewhere
        mean, std = get_normalization_config(norm_config_idx, actual_device)
        # get_frame_transforms will include ToDtype, Resize, Normalize.
        # These transforms (esp. Normalize) expect CHW tensors.
        frame_transforms_comp = get_frame_transforms(use_half_precision, mean, std, vr_video, img_size)

        for i in range(0, len(vr), frames_to_skip):
            try:
                # vr[i] directly returns a torch.Tensor
                frame = vr[i].to(actual_device) # HWC, RGB, (likely) uint8 tensor on device
            except RuntimeError as e_read_frame:
                logger.warning(f"Could not read frame {i} from {video_path}: {e_read_frame}")
                continue

            if vr_video:
                # Assuming vr_permute returns HWC, or format suitable for permute(2,0,1) if needed
                frame = vr_permute(frame)

            # Permute HWC to CHW for standard torchvision transforms
            if frame.ndim == 3: # Ensure it's an image tensor (H, W, C)
                 frame = frame.permute(2, 0, 1) # Convert to (C, H, W)
            # else: frame might not be a 3D tensor, or vr_permute changed it. Let transforms handle or error.

            transformed_frame = frame_transforms_comp(frame) 
            
            frame_identifier: Union[int, float] = i / fps if use_timestamps else i
            yield (frame_identifier, transformed_frame)
            
    if 'vr' in locals(): del vr

def crop_black_bars_lr(frame: torch.Tensor, black_threshold: float = 10.0, column_black_pixel_fraction_threshold: float = 0.95) -> torch.Tensor:
    """Crops vertical black bars from the left and right of an HWC image tensor."""
    logger = logging.getLogger("logger")
    if not isinstance(frame, torch.Tensor) or frame.ndim != 3 or frame.shape[2] < 3: # Expect HWC, at least 3 channels
        logger.warning(f"crop_black_bars_lr: Invalid frame shape {frame.shape if isinstance(frame, torch.Tensor) else type(frame)}, returning original frame.")
        return frame

    H, W, C = frame.shape
    if W == 0 or H == 0:
        logger.debug("crop_black_bars_lr: Frame has zero width or height, returning original frame.")
        return frame

    # Consider only RGB for blackness detection, works even if C > 3 (e.g., RGBA)
    rgb_frame = frame[:, :, :3]

    # Boolean tensor: True where pixels are "black"
    # A pixel is black if all its RGB components are below the threshold
    is_black_pixel = torch.all(rgb_frame < black_threshold, dim=2) # Shape (H, W)

    # Fraction of black pixels in each column
    column_black_pixel_count = torch.sum(is_black_pixel, dim=0) # Shape (W)
    column_black_fraction = column_black_pixel_count.float() / H # Shape (W)

    # Identify columns that are considered part of a black bar
    is_black_bar_column = column_black_fraction >= column_black_pixel_fraction_threshold # Shape (W)

    x_start = 0
    # Find the first non-black-bar column from the left
    for i in range(W):
        if not is_black_bar_column[i]:
            x_start = i
            break
    else: # All columns are black bars, or frame is effectively empty
        logger.debug("crop_black_bars_lr: Frame appears to be entirely black or too narrow. No crop applied.")
        return frame

    x_end = W
    # Find the first non-black-bar column from the right (working backwards)
    for i in range(W - 1, x_start -1, -1): # Iterate from right to left, but not past x_start
        if not is_black_bar_column[i]:
            x_end = i + 1 # Slice is exclusive at the end
            break
    
    if x_start >= x_end: # Should not happen if the first loop found content
        logger.warning(f"crop_black_bars_lr: Inconsistent crop boundaries (x_start={x_start}, x_end={x_end}). No crop applied.")
        return frame
    
    if x_start == 0 and x_end == W: # No bars detected
        # logger.debug("crop_black_bars_lr: No black bars detected to crop.")
        return frame

    cropped_frame = frame[:, x_start:x_end, :]
    logger.debug(f"Cropped frame from W={W} to W'={cropped_frame.shape[1]} (x_start={x_start}, x_end={x_end})")
    return cropped_frame
    