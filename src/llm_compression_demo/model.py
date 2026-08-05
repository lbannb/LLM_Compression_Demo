import json, torch

from huggingface_hub import hf_hub_download
from safetensors import safe_open

# docs on huggingface: https://huggingface.co/docs/transformers/en/model_doc/llama2
REPO  = "meta-llama/Llama-2-7b-hf"  # what CompactifAI benchmarks
INDEX = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"] # stores the mapping of weight names to their corresponding safetensors files

""" 
pt  / torch  -> torch.Tensor
np  / numpy  -> numpy.ndarray
tf, flax, mlx -> need tensorflow / jax / mlx
"""
TENSOR_DTYPE = "pt"


# As I have a RTX 2080 Ti with 11GB VRAM, I cannot load the model in float16 precision on the GPU :(
# As we want to demonstrate compression anyway, we will laod only he individual weight matrices layer by layer, such
# that in the end the fully compressed model should fot on my GPU

"""
    Returns the projection matrices of the self-attention layer at index layer_indx in the model m.
    The returned dictionary has keys "q_proj", "k_proj", "v_proj", and "o_proj" corresponding to the query, key, value, and output projection matrices, respectively.
"""

def get_decoder_layer(layer_indx: int, type:str = "self_attn", device:str = "cuda") -> dict[str, torch.Tensor]:
    assert type in ["self_attn", "mlp"], f"Invalid type {type}. Must be 'self_attn' or 'mlp'."

    # filters all weight matrices from the self_attn layer
    prefix = f"model.layers.{layer_indx}.{type}."
    keys = [k for k in INDEX if k.startswith(prefix) and k.endswith("_proj.weight")]

    # open the safetensors file corresponding to the first key in the list and load the projection matrices into a dictionary
    with safe_open(hf_hub_download(REPO, INDEX[keys[0]]), framework=TENSOR_DTYPE, device=device) as f:
        proj_mat_name = lambda key: key[len(prefix):-len(".weight")] # q_proj, k_proj, v_proj, o_proj for self_attn, up_proj, down_proj, gate_proj for mlp
        return {proj_mat_name(k): f.get_tensor(k).float() for k in keys}
