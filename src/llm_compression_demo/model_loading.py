import json, torch

from huggingface_hub import hf_hub_download
from safetensors import safe_open

# docs on huggingface: https://huggingface.co/docs/transformers/en/model_doc/llama2
REPO  = "meta-llama/Llama-2-7b-hf"  # what CompactifAI benchmarks
DEVICE = "cuda"  # or "cpu"
INDEX = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"] # stores the mapping of weight names to their corresponding safetensors files
TOTAL_SIZE  = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["metadata"]["total_size"]
EMBED_TOKENS_INDEX = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]["model.embed_tokens.weight"]

""" 
pt  / torch  -> torch.Tensor
np  / numpy  -> numpy.ndarray
tf, flax, mlx -> need tensorflow / jax / mlx
"""
FRAMEWORK = "pt"

# As I have a RTX 2080 Ti with 11GB VRAM, I cannot load the model in float16 precision on the GPU :(
# As we want to demonstrate compression anyway, we will load only the individual weight matrices layer by layer, such
# that in the end the fully compressed model should fit on my GPU

def get_decoder_layer(layer_indx: int, layer_type:str = "self_attn", device:str = DEVICE) -> dict[str, torch.Tensor]:
    """
        Returns the projection matrices of the self-attention layer at index layer_indx in the model m.
        The returned dictionary has keys "q_proj", "k_proj", "v_proj", and "o_proj" corresponding to the query, key, value, and output projection matrices, respectively.
    """

    assert layer_type in ["self_attn", "mlp"], f"Invalid layer_type {layer_type}. Must be 'self_attn' or 'mlp'."

    # filters all weight matrices from the self_attn layer
    prefix = f"model.layers.{layer_indx}.{layer_type}."
    keys = [k for k in INDEX if k.startswith(prefix) and k.endswith("_proj.weight")]

    # open the safetensors file corresponding to the first key in the list and load the projection matrices into a dictionary
    with safe_open(hf_hub_download(REPO, INDEX[keys[0]]), framework=FRAMEWORK, device=device) as f:
        proj_mat_name = lambda key: key[len(prefix):-len(".weight")] # q_proj, k_proj, v_proj, o_proj for self_attn, up_proj, down_proj, gate_proj for mlp
        return {proj_mat_name(k): f.get_tensor(k).float() for k in keys}

def get_embed_tokens(device: str = DEVICE) -> torch.Tensor:
    """
        Returns the token embedding table, shape [vocab_size, hidden_dim] = [32000, 4096] for Llama2-7b.
        The forward pass is a row lookup x = E[token_ids], not a matmul, so this matrix is not a compression target.
    """
    key = "model.embed_tokens.weight"
    with safe_open(hf_hub_download(REPO, INDEX[key]), framework=FRAMEWORK, device=device) as f:
        return f.get_tensor(key).float()

def get_number_of_layers() -> int:
    return len({int(k.split(".layers.")[1].split(".")[0]) for k in INDEX if ".layers." in k})

def print_model_info():
    n_layers = get_number_of_layers()

    # get decoder block information
    sa_layer = get_decoder_layer(0)
    mlp_layer = get_decoder_layer(0, layer_type="mlp")

    assert sa_layer and mlp_layer, "no attention/MLP weights found"

    n_params = n_layers *(sum(layer.numel() for layer in sa_layer.values()) + sum(layer.numel() for layer in mlp_layer.values()))
    n_matrices = n_layers * (len(sa_layer) + len(mlp_layer))

    print(f"{REPO}: {TOTAL_SIZE / 1e9:.2f}GB, "
          f"{n_params / 1e9:.2f}B tensorizable params in {n_matrices} matrices")