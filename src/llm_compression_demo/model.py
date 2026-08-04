import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PAPER_MODEL = "meta-llama/Llama-2-7b-hf"  # what CompactifAI benchmarks
MODEL_ID = os.environ.get("LLM_DEMO_MODEL", "meta-llama/Llama-2-7b-hf") # set env var LLM_DEMO_MODEL to "meta-llama/Llama-2-7b-chat-hf"

# loads the model in float16, so we can run the code locally
def load(model_id: str = MODEL_ID, quantization: torch.dtype = torch.float16):
    """Returns (model, tokenizer). Downloads on first call, cached after."""
    return (
        # choose head type from: https://huggingface.co/docs/transformers/en/models. Here, Llama-2-7b-hf is a causal LM, so we use AutoModelForCausalLM. For other model types, see the docs.
        AutoModelForCausalLM.from_pretrained(model_id, dtype=quantization), # Call from_pretrained() to download and load a model’s weights and configuration stored on the Hugging Face Hub.
        AutoTokenizer.from_pretrained(model_id), # load tokenizer
    )
