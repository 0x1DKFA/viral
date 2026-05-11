import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import json
import re

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
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def analyze(self, video_path):
        # Construct the prompt for viral highlight detection
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 1.0, # Sample 1 frame per second to save tokens/memory
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
