import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
from io import BytesIO
from PIL import Image
import logging
import os # Added for reading tag_list_path
import random # Added for jitter
import time   # Added for custom sleep
from typing import Dict, Any, Optional, List, Tuple, TextIO # Added imports

# Custom Retry class with jitter
class RetryWithJitter(Retry):
    def __init__(self, *args: Any, jitter_factor: float = 0.25, **kwargs: Any): # Added types for args, jitter_factor, kwargs
        super().__init__(*args, **kwargs)
        self.jitter_factor: float = jitter_factor
        if not (0 <= self.jitter_factor <= 1):
            # Warn if jitter_factor is outside typical range, but allow it.
            # Depending on use case, >1 might be desired for very aggressive jitter.
            logging.getLogger("logger").warning(
                f"RetryWithJitter initialized with jitter_factor={self.jitter_factor}, which is outside the typical [0, 1] range."
            )

    def sleep(self, backoff_value: float) -> None: # Added type for backoff_value and return type
        """Sleep for the backoff time, adding jitter."""
        # Respect Retry-After header if present (behavior from parent)
        retry_after: Optional[float] = self.get_retry_after(response=self._last_response)
        if retry_after:
            time.sleep(retry_after)
            return

        # Calculate jitter: random percentage of backoff_value up to jitter_factor
        # This adds jitter on top of the exponentially backed-off value.
        jitter: float = random.uniform(0, backoff_value * self.jitter_factor)
        sleep_duration: float = backoff_value + jitter
        
        # Ensure sleep duration is not negative, then sleep
        time.sleep(max(0, sleep_duration))

class OpenAICompatibleVLMClient:
    def __init__(
        self,
        config: Dict[str, Any], # Expects a dictionary with API and model settings
    ):
        # Load parameters from config
        self.api_base_url: str = str(config["api_base_url"]).rstrip('/')
        self.model_id: str = str(config["model_id"])
        self.max_new_tokens: int = int(config.get("max_new_tokens", 128))
        self.request_timeout: int = int(config.get("request_timeout", 70)) # seconds
        self.vlm_detected_tag_confidence: float = float(config.get("vlm_detected_tag_confidence", 0.99))
        
        tag_list_path: Optional[str] = config.get("tag_list_path")
        self.tag_list: List[str]
        if tag_list_path and os.path.exists(tag_list_path):
            with open(tag_list_path, 'r') as f:
                f_typed: TextIO = f # For type clarity
                self.tag_list = [line.strip() for line in f_typed if line.strip()]
        elif "tag_list" in config and isinstance(config["tag_list"], list):
             self.tag_list = config["tag_list"]
        else:
            raise ValueError("Configuration must provide 'tag_list_path' or a 'tag_list'.")
        assert self.tag_list, "Loaded tag list is empty."

        self.logger: logging.Logger = logging.getLogger("logger")

        # Setup retry mechanism for requests
        retry_attempts: int = int(config.get("retry_attempts", 3))
        retry_backoff_factor: float = float(config.get("retry_backoff_factor", 0.5))
        retry_jitter_factor: float = float(config.get("retry_jitter_factor", 0.25)) # New config for jitter
        status_forcelist: Tuple[int, ...] = (500, 502, 503, 504) # Common server-side errors to retry

        # Use the custom RetryWithJitter class
        retry_strategy: RetryWithJitter = RetryWithJitter(
            total=retry_attempts,
            backoff_factor=retry_backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["POST"], # Ensure POST requests are retried
            respect_retry_after_header=True,
            jitter_factor=retry_jitter_factor # Pass jitter factor
        )
        adapter: HTTPAdapter = HTTPAdapter(max_retries=retry_strategy)
        self.session: requests.Session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.logger.info(
            f"Initializing OpenAICompatibleVLMClient for model {self.model_id} "
            f"with {len(self.tag_list)} tags, targeting API: {self.api_base_url}. "
            f"Retry: {retry_attempts} attempts, backoff {retry_backoff_factor}s, jitter factor {retry_jitter_factor}."
        )
        self.logger.info(f"OpenAI VLM client initialized successfully")

    def _convert_image_to_base64_data_url(self, frame: Image.Image, format: str = "JPEG") -> str:
        buffered: BytesIO = BytesIO()
        frame.save(buffered, format=format)
        img_str: str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"

    def analyze_frame(self, frame: Optional[Image.Image]) -> Dict[str, float]: # Added Optional to frame
        """
        Process one frame:
          1) convert image to base64
          2) build API payload
          3) run inference via API call
          4) parse tags + assign confidences
        """
        tag: str # for dict comprehension
        if not frame:
            self.logger.warning("Analyze_frame called with no frame.")
            return {tag: 0.0 for tag in self.tag_list}

        try:
            image_data_url: str = self._convert_image_to_base64_data_url(frame)
        except Exception as e_convert: # Renamed exception
            self.logger.error(f"Failed to convert image to base64: {e_convert}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list}

        tags_str: str = ", ".join(self.tag_list)
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {
                        "type": "text",
                        "text": (
                            f"What is happening in this scene?"
                        ),
                    },
                ],
            }
        ]

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0, # For deterministic output, as in do_sample=False
            "stream": False,
        }

        api_url: str = f"{self.api_base_url}/v1/chat/completions"
        self.logger.debug(f"Sending request to {self.model_id} at {api_url} with image and {len(self.tag_list)} tags.")
        raw_reply: str = ""
        try:
            # Use the session with retry logic
            response: requests.Response = self.session.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            
            response_data: Dict[str, Any] = response.json()
            if response_data.get("choices") and response_data["choices"][0].get("message"):
                raw_reply = response_data["choices"][0]["message"].get("content", "")
            else:
                self.logger.error(f"Unexpected response structure from API: {response_data}")
                return {tag: 0.0 for tag in self.tag_list}

            self.logger.debug(f"Response received from {self.model_id}: {raw_reply}")

        except requests.exceptions.RequestException as e_req: # Renamed exception
            self.logger.error(f"API request to {api_url} failed: {e_req}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list} # Return all tags with 0 confidence on error
        except Exception as e_general: # Renamed exception
            self.logger.error(f"An unexpected error occurred during API call or response processing: {e_general}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list}

        return self._parse_simple_default(raw_reply)

    def _parse_simple_default(self, reply: str) -> Dict[str, float]:
        # Initialize all configured tags (from self.tag_list, which preserves original casing) with 0.0 confidence
        found: Dict[str, float] = {tag: 0.0 for tag in self.tag_list}
        
        # Split the VLM's reply into individual tags and convert to lowercase for matching
        parsed_vlm_tags: List[str] = [tag.strip().lower() for tag in reply.split(',') if tag.strip()]

        # For each tag in our configured list (self.tag_list preserves original casing)
        for tag_config_original_case in self.tag_list:
            # If the lowercase version of the configured tag is in the lowercase list of VLM-returned tags
            if tag_config_original_case.lower() in parsed_vlm_tags:
                # Use the original casing from self.tag_list as the key in the 'found' dictionary
                found[tag_config_original_case] = self.vlm_detected_tag_confidence
        
        return found

    def _parse_score_per_line(self, reply: str) -> Dict[str, float]:
        # Implementation of _parse_score_per_line method
        pass
