import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import json
import re
import cv2
from PIL import Image

class VLMAnalyzer:
    def __init__(self, model_id="Qwen/Qwen3-VL-8B-Instruct"):
        print(f"Loading model {model_id} in 4-bit...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def _extract_frames(self, video_path, fps=1.0):
        """Extract frames manually using OpenCV to bypass broken torchvision.io."""
        container = cv2.VideoCapture(video_path)
        video_fps = container.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30 # Fallback
            
        step = max(1, int(video_fps / fps))
        frames = []
        count = 0
        while True:
            ret, frame = container.read()
            if not ret:
                break
            if count % step == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            count += 1
        container.release()
        return frames

    def analyze(self, video_path):
        # Extract frames manually to bypass torchvision bug
        frames = self._extract_frames(video_path, fps=1.0)
        
        if not frames:
            print(f"Warning: No frames extracted from {video_path}")
            return {"score": 0, "action_center_x": 0.5}

        # Construct the prompt for viral highlight detection
        # We pass the frames as a 'video' content type, but providing the PIL images directly
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames, # Pass list of PIL images
                        "fps": 1.0,
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze this gameplay clip for viral potential. "
                            "Rate it on a scale of 1-10 (10 being extremely exciting/highlight worthy). "
                            "Also identify the horizontal center of the action as a percentage (0.0 to 1.0, where 0.5 is the middle). "
                            "Return ONLY a JSON object: {'score': int, 'action_center_x': float}"
                        ),
                    },
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse JSON from output
        try:
            # Find JSON block in case the model added extra text
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                print(f"Failed to find JSON in output: {output_text}")
                return {"score": 0, "action_center_x": 0.5}
        except Exception as e:
            print(f"Error parsing VLM output: {e}. Output was: {output_text}")
            return {"score": 0, "action_center_x": 0.5}
