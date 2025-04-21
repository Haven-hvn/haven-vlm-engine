import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import logging

class VLMModel:
    def __init__(
        self,
        tag_list: list[str],
        model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
        use_quantization: bool = True,
        device: str = None,
        max_new_tokens: int = 128,
    ):
        # Device & settings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        assert tag_list, "Provide a non-empty list of NSFW tags."
        self.tag_list = tag_list
        self.logger = logging.getLogger("logger")

        self.logger.info(f"Initializing VLM model with {len(tag_list)} tags on {self.device}")
        
        # Load 4-bit quantized VLM
        bnb_config = BitsAndBytesConfig(load_in_4bit=use_quantization)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
        ).to(self.device)

        # Processor handles resizing, normalization, chat‐template
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.logger.info(f"VLM model initialized successfully")

    def _format_prompt(self, frame: Image.Image) -> str:
        """
        Build a chat‐style multimodal prompt:
          - one image
          - instruction listing all tags
        """
        tags_str = ", ".join(self.tag_list)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
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
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

    def analyze_frame(self, frame: Image.Image) -> dict[str, float]:
        """
        Process one frame:
          1) build prompt
          2) run inference
          3) parse tags + assign confidences
        """
        prompt = self._format_prompt(frame)
        inputs = self.processor(
            text=prompt,
            images=[frame],
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        raw_reply = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return self._parse_simple_default(raw_reply)

    def _parse_simple_default(self, reply: str) -> dict[str, float]:
        """
        Simple Default Scheme:
          - Any tag listed by the model → confidence = 0.8
          - All other tags → confidence = 0.0
        """
        # Remove any "Assistant:" prefix
        if ":" in reply:
            reply = reply.split(":", 1)[1]
        # Split on commas and normalize
        found = {tag: 0.0 for tag in self.tag_list}
        for chunk in reply.split(","):
            t = chunk.strip().lower()
            for tag in self.tag_list:
                if tag.lower() == t:
                    found[tag] = 0.8
        return found
