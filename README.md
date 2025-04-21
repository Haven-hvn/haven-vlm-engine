# HAVEN VLM Engine: Advanced Content Tagging System

This project introduces an advanced Vision-Language Model (VLM) implementation for automatic content tagging, delivering superior accuracy compared to traditional image classification methods.

## Key Advantages

* **Context-Aware Detection**: Leverages VLM understanding of visual relationships for more accurate tagging
* **Comprehensive Tag Support**: Available depending on VLM-finetune and versioning

## Implementation Details

### Core Architecture

#### VLM Model (`lib/model/vlm_model.py`)
The foundation of the system handles:
1. Loading 4-bit quantized pre-trained vision-language models
2. Intelligent prompt formatting combining images with tag lists
3. Response parsing to extract relevant tags
4. Confidence score assignment (default 0.8 for detected tags)

#### Integration Components
1. **Direct AI Model** (`lib/model/vlm_ai_model.py`):
   - Inherits from base `Model` class
   - Manages VLM model loading and tag list
   - Processes frames into pipeline-compatible outputs

2. **Python Function** (`lib/model/python_functions.py`):
   - Provides `vlm_frame_analyzer` function
   - Alternative integration path for pipeline use

### Configuration System

- **Model Configs**:
  - `config/models/vlm_model.yaml`: Primary VLM model settings
  - `config/models/vlm_frame_analyzer.yaml`: Python function configuration

- **Pipeline Setup**:
  - `config/pipelines/vlm_pipeline.yaml`: Complete processing pipeline

- **Tag Management**:
  - `models/tags.txt`: Configurable tag definitions

## Limitations

- **Hardware Requirements**: Only NVIDIA GPUs are supported. Any NVIDIA GPUs older than the NVIDIA 1080 will likely not work. Support for AMD GPUs is not planned.

- **Performance on CPU**: While CPU is supported, it will be much slower compared to running on a GPU.

- **Complexity and Support**: Running machine learning models is complex and requires precise conditions to work smoothly. Although we have worked to make the installation process and AI model predictions as user-friendly as possible, due to the variability in hardware and software environments, there might be issues. We will do our best to help resolve any issues, but we cannot guarantee that the models will run on every computer.

## Getting Started

### Installation
To get started with the HAVEN VLM Engine, first install the required dependencies:
```bash
pip install -r requirements.txt
```
### Basic Integration
1. Add the VLM model to `config/active_ai.yaml`:
```yaml
active_ai_models:
  - vlm_nsfw_model
```

2. Run your pipeline to automatically utilize the VLM model.

### Direct Usage Example
Use the provided example script for direct model interaction:
```python
from lib.model.vlm_model import process_video_with_vlm

tag_list = load_tag_list("path/to/tags.txt")
video_result, video_tag_info = process_video_with_vlm(video_path, tag_list)

print("Detected Tags:", video_tag_info.video_tags)
print("Tag Timespans:", video_tag_info.tag_timespans)
```

## Customization

### Model Selection
*   Modify `vlm_model_name` in `config/models/vlm_nsfw_model.yaml` to change the VLM model.

### Tag Management
*   Edit `models/tags.txt` to:
    *   Add new tags
    *   Remove unnecessary tags
    *   Adjust tag phrasing

### Confidence Thresholds
*   Adjust default confidence scores by modifying `_parse_simple_default` in `lib/model/vlm_model.py`.

---

## Performance Considerations

*   **Hardware:** NVIDIA GPU with CUDA recommended.
*   **Quantization:** 4-bit quantization reduces memory usage.
*   **Speed/Accuracy Tradeoffs:** Larger models offer better accuracy but slower processing.

---

## Support and Community

*   **GitHub:** Report issues at the [HAVEN VLM Engine repository](https://github.com/Haven-hvn/haven-vlm-engine)

---

## Future Developments

Planned features include:
*   Expanded tag categories
*   Improved quantization techniques
*   Simplified configuration tools
