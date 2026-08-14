import torch

from  pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

import llm_compression_demo.model_loading as m
import llm_compression_demo.compression as c

if __name__ == "__main__":

    CHI = 20
    DIMS_SA = [8,8,8,8]
    DIMS_MLP = [8,8,4,43]

    for layer_indx in range(m.get_number_of_layers()):
        print("--------------------------------------------------------")
        print(f"Compressing layer {layer_indx} with chi={CHI}...")

        sa_layer = m.get_decoder_layer(layer_indx, layer_type="self_attn")
        mlp_layer = m.get_decoder_layer(layer_indx, layer_type="mlp")

        # check if compressed weight matrices are already stored in the local folder
        dir = Path(__file__).parent.parent.parent / "data" / "compressed_weights" / f"chi={CHI}"
        dir.mkdir(exist_ok=True)

        sa_mpo, mlp_mpo = {}, {}

        # compress the weight matrices of the self-attention layer
        for proj_mat_name, tensor in sa_layer.items():
            if not (dir / f"layer_{layer_indx}_{proj_mat_name}.pt").exists():
                print(f"Compressing {proj_mat_name}...")

                sa_mpo = c.compress_tensor(tensor, CHI, DIMS_SA, DIMS_SA)
                save_file(c.mpo_train_to_safetensor_dict(sa_mpo), dir / f"layer_{layer_indx}_{proj_mat_name}.pt")
            else:
                print(f"{proj_mat_name} is already compressed. Skipped compression.")

                sa_mpo = safe_open(dir / f"layer_{layer_indx}_{proj_mat_name}.pt", framework="pt", device="cpu")

        # compress the weight matrices of the mlp layer
        for proj_mat_name, tensor in mlp_layer.items():
            if not (dir / f"layer_{layer_indx}_{proj_mat_name}.pt").exists():
                print(f"Compressing {proj_mat_name}...")

                # gate_proj and up_proj are matrices of shape (11008, 4096) and down_proj is a matrix of shape (4096, 11008)
                mlp_mpo = c.compress_tensor(tensor, CHI, DIMS_SA, DIMS_MLP) if proj_mat_name == "down_proj" else c.compress_tensor(tensor, CHI, DIMS_MLP, DIMS_SA)
                save_file(c.mpo_train_to_safetensor_dict(mlp_mpo), dir / f"layer_{layer_indx}_{proj_mat_name}.pt")
            else:
                print(f"{proj_mat_name} is already compressed. Skipped compression.")

                mlp_mpo = safe_open(dir / f"layer_{layer_indx}_{proj_mat_name}.pt", framework="pt", device="cpu")

        print()
