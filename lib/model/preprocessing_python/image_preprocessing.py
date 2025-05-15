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

def get_normalization_config(index: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    normalization_configs: List[Tuple[torch.Tensor, torch.Tensor]] = [
        (torch.tensor([0.485, 0.456, 0.406], device=device), torch.tensor([0.229, 0.224, 0.225], device=device)),
        (torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device), torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)),
    ]
    if 0 <= index < len(normalization_configs):
        return normalization_configs[index]
    else:
        logging.getLogger("logger").warning(f"Invalid normalization config index: {index}. Falling back to index 0.")
        return normalization_configs[0]

def custom_round(x: float, base: int = 1) -> int:
    return int(base * round(x/base))

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

def get_frame_transforms(
    use_half_precision: bool, 
    mean: torch.Tensor, 
    std: torch.Tensor, 
    vr_video: bool = False, 
    img_size: Union[int, Tuple[int,int]] = 512
) -> transforms.Compose:
    target_dtype: torch.dtype = torch.float16 if use_half_precision else torch.float32
    transform_list: List[Any] = []
    transform_list.append(transforms.ToDtype(target_dtype, scale=True))
    
    current_img_size: Union[int, Tuple[int,int]]
    if isinstance(img_size, int):
        current_img_size = (img_size, img_size)
    else:
        current_img_size = img_size # It's already a tuple

    # Apply resize if img_size is provided.
    if img_size is not None: # Check if img_size itself is None, not current_img_size
        transform_list.append(transforms.Resize(current_img_size, interpolation=InterpolationMode.BICUBIC, antialias=True))

    transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)

def vr_permute(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim != 3 or frame.shape[2] != 3:
        logging.getLogger("logger").warning(f"vr_permute received unexpected frame shape: {frame.shape}")
        return frame

    height: int = frame.shape[0]
    width: int = frame.shape[1]
    aspect_ratio: float = width / height if height > 0 else 0

    if aspect_ratio > 1.5: 
        return frame[:, :width//2, :] 
    else: 
        return frame[:height//2, width//4:3*width//4, :]

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
    
def preprocess_video(video_path: str, frame_interval_sec: float = 0.5, img_size: Union[int, Tuple[int,int]] = 512, use_half_precision: bool = True, device_str: Optional[str] = None, use_timestamps: bool = False, vr_video: bool = False, norm_config_idx: int = 1) -> Iterator[Tuple[Union[int, float], torch.Tensor]]:
    actual_device: torch.device = torch.device(device_str) if device_str else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = get_normalization_config(norm_config_idx, actual_device)
    
    frame_transforms_comp: transforms.Compose = get_frame_transforms(use_half_precision, mean, std, vr_video, img_size)
    
    vr: decord.VideoReader
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    except RuntimeError as e:
        logging.getLogger("logger").error(f"Decord failed to open video {video_path}: {e}")
        return
        
    fps: float = vr.get_avg_fps()
    if fps == 0:
        logging.getLogger("logger").warning(f"Video {video_path} has FPS of 0. Cannot process.")
        del vr
        return

    frames_to_skip: int = custom_round(fps * frame_interval_sec)
    if frames_to_skip < 1: frames_to_skip = 1

    i: int
    for i in range(0, len(vr), frames_to_skip):
        try:
            frame: torch.Tensor = vr[i]
        except RuntimeError as e_read_frame:
            logging.getLogger("logger").warning(f"Could not read frame {i} from {video_path}: {e_read_frame}")
            continue

        frame = frame.to(actual_device) 
        
        if vr_video:
            frame = vr_permute(frame) 
        
        frame = frame.permute(2, 0, 1) 
        
        transformed_frame: torch.Tensor = frame_transforms_comp(frame)
        
        frame_identifier: Union[int, float] = i / fps if use_timestamps else i
            
        yield (frame_identifier, transformed_frame)
    del vr
    