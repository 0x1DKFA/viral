import os
import pytest
from src.analyzer import VLMAnalyzer
from dotenv import load_dotenv

load_dotenv()

@pytest.mark.skip(reason="Downloads 7B model and requires GPU, run manually for smoke test")
def test_analyzer_load():
    model_id = os.getenv("VLM_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
    analyzer = VLMAnalyzer(model_id=model_id)
    assert analyzer.model is not None
    assert analyzer.processor is not None
