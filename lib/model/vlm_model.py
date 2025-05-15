import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
from io import BytesIO
from PIL import Image
import logging
import os # Added for reading tag_list_path

class OpenAICompatibleVLMClient:
    def __init__(
        self,
        config: dict, # Expects a dictionary with API and model settings
    ):
        # Load parameters from config
        self.api_base_url = str(config["api_base_url"]).rstrip('/')
        self.model_id = str(config["model_id"])
        self.max_new_tokens = int(config.get("max_new_tokens", 128))
        self.request_timeout = int(config.get("request_timeout", 70)) # seconds
        
        tag_list_path = config.get("tag_list_path")
        if tag_list_path and os.path.exists(tag_list_path):
            with open(tag_list_path, 'r') as f:
                self.tag_list = [line.strip() for line in f if line.strip()]
        elif "tag_list" in config and isinstance(config["tag_list"], list):
             self.tag_list = config["tag_list"]
        else:
            raise ValueError("Configuration must provide 'tag_list_path' or a 'tag_list'.")
        assert self.tag_list, "Loaded tag list is empty."

        self.logger = logging.getLogger("logger")

        # Setup retry mechanism for requests
        retry_attempts = int(config.get("retry_attempts", 3))
        retry_backoff_factor = float(config.get("retry_backoff_factor", 0.5))
        status_forcelist = (500, 502, 503, 504) # Common server-side errors to retry

        retry_strategy = Retry(
            total=retry_attempts,
            backoff_factor=retry_backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["POST"], # Ensure POST requests are retried
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.logger.info(
            f"Initializing OpenAICompatibleVLMClient for model {self.model_id} "
            f"with {len(self.tag_list)} tags, targeting API: {self.api_base_url}. "
            f"Retry: {retry_attempts} attempts, backoff {retry_backoff_factor}s."
        )
        self.logger.info(f"OpenAI VLM client initialized successfully")

    def _convert_image_to_base64_data_url(self, frame: Image.Image, format: str = "JPEG") -> str:
        buffered = BytesIO()
        frame.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"

    def analyze_frame(self, frame: Image.Image) -> dict[str, float]:
        """
        Process one frame:
          1) convert image to base64
          2) build API payload
          3) run inference via API call
          4) parse tags + assign confidences
        """
        if not frame:
            self.logger.warning("Analyze_frame called with no frame.")
            return {tag: 0.0 for tag in self.tag_list}

        try:
            image_data_url = self._convert_image_to_base64_data_url(frame)
        except Exception as e:
            self.logger.error(f"Failed to convert image to base64: {e}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list}

        tags_str = ", ".join(self.tag_list)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {
                        "type": "text",
                        "text": (
                            f"Identify which of these NSFW tags apply: {tags_str}. "
                            "Reply **only** with the tags you see, separated by commas."
                        ),
                    },
                ],
            }
        ]

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0, # For deterministic output, as in do_sample=False
            "stream": False,
        }

        api_url = f"{self.api_base_url}/v1/chat/completions"
        self.logger.debug(f"Sending request to {self.model_id} at {api_url} with image and {len(self.tag_list)} tags.")

        raw_reply = ""
        try:
            # Use the session with retry logic
            response = self.session.post(
                api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout,
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            
            response_data = response.json()
            if response_data.get("choices") and response_data["choices"][0].get("message"):
                raw_reply = response_data["choices"][0]["message"].get("content", "")
            else:
                self.logger.error(f"Unexpected response structure from API: {response_data}")
                return {tag: 0.0 for tag in self.tag_list}

            self.logger.debug(f"Response received from {self.model_id}: {raw_reply}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request to {api_url} failed: {e}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list} # Return all tags with 0 confidence on error
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during API call or response processing: {e}", exc_info=True)
            return {tag: 0.0 for tag in self.tag_list}

        return self._parse_simple_default(raw_reply)

    def _parse_simple_default(self, reply: str) -> dict[str, float]:
        """
        Simple Default Scheme:
          - Any tag listed by the model → confidence = 0.8
          - All other tags → confidence = 0.0
        """
        # Remove any "Assistant:" prefix or similar, an LLaMA model might add this.
        # The API spec for OpenAI usually returns only the content.
        # Let's ensure the reply is a string first.
        if not isinstance(reply, str):
            self.logger.warning(f"_parse_simple_default received non-string reply: {type(reply)}")
            return {tag: 0.0 for tag in self.tag_list}

        # Check for common prefixes if any (though OpenAI API usually clean)
        if ":" in reply:
            parts = reply.split(":", 1)
            if len(parts) > 1 and parts[0].lower().strip() in ["assistant", "model", "ai", "bot", "response", "reply"]:
                 reply = parts[1]
        
        # Split on commas and normalize
        found = {tag: 0.0 for tag in self.tag_list}
        parsed_tags = [t.strip().lower() for t in reply.split(",") if t.strip()]
        
        for t_model in parsed_tags:
            for tag_config in self.tag_list:
                if tag_config.lower() == t_model:
                    found[tag_config] = 0.8 # Default confidence
                    break # Move to next model tag once matched
        return found
