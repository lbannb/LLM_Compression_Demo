import math

from pathlib import Path
from safetensors.torch import save_file

import llm_compression_demo.model_loading as m
import llm_compression_demo.compression as c

# My choice of bond dim. for the sa layer and mlp layer
DIMS_SA = [8, 8, 8, 8]
DIMS_MLP = [8, 8, 4, 43]

"""
    According to the CompactifAI paper ( supplementary, arXiv 2401.14109, Fig. 5), the first few layers are more sensitive to compression than the later layers. 
    They state:

    - "the layers at the beginning should be handled with more care and it is advised not to compress them below 50%". 
      50% of a 4096x4096 means with our choice of `DIMS_SA` and `DIMS_MLP`:

      bond_1 = min(chi, 8^2) = min(chi, 64)
      bond_2 = min(chi, 8^4) = min(chi, 4096)
      bond_3 = min(chi, 8^2) = min(chi, 64)

      To get a compression of 50% we calculate:

      compressed_model = 8^2*64 + 64*8^2*chi + chi*8^2*64 + 64 * 8^2 = 4096 + 4096*chi + 4096*chi + 4096 = 8192*(chi+1)
      uncompressed_model = 4096*4096

      ratio = compressed_model / uncompressed_model = 8192*(chi+1) / (4096*4096) = 2*(chi+1)/4096
      ratio = 0.5 => 2*(chi+1)/4096 = 0.5 => chi+1 = 1024 => chi = 1023 ~ 1024
       
    - for the MLP layer, which is 11008x4096, we have:

      bond_1 = min(chi, 8^2) = min(chi, 64)
      bond_2 = min(chi, 8^4) = min(chi, 4096)
      bond_3 = min(chi, 43*8) = min(chi, 344)

      compressed_model = 8^2*64 + 64*8^2*chi + chi*8*4*344 + 344 * 8 * 43 = 4096 + 4096*chi + 11008*chi + 118336 = 15104 * chi + 122432
      uncompressed_model = 11008*4096

      ratio = compressed_model / uncompressed_model = (15104 * chi + 122432) / (11008*4096)
      ratio = 0.5 => (15104 * chi + 122432) / (11008*4096) = 0.5 => 15104 * chi + 122432 = 22016·1024 => chi = (22016*1024 - 122432) / 15104 = 1484.5 ~ 1485

    - Block[15] and Block[31] are flat across the whole chi_max range, and the text puts the robust blocks at "10% of the original size" 
      same calculation as above then gives:

      sa-layer: 2*(chi+1)/4096 = 0.1 => chi + 1 = 204.8 => chi ~ 204
      mlp-layer: (15104 * chi + 122432) / (11008*4096) = 0.1 => 15104 * chi + 122432 = 11008*409.6 => chi = (11008*409.6 - 122432) / 15104 = 312.8 ~ 313

"""
SCHEDULE = ((2, 1024), (6, 512), (16, 160), (32, 64))  # (bound, chi): blocks below the bound get that chi. Tiers chosen so the whole model lands on the paper's 2.1 B parameters (68.9% reduction).
EXCLUDED = ("down_proj",) # down_proj is the output of each block, and the paper advises to leave it dense. (See Fig. 5 of the CompactifAI supplementary, arXiv 2401.14109.)

def chi_for(block: int, proj_name: str) -> int | None:
    """Returns the bond dimension for one projection matrix according to the `SCHEDULE`, or `None` if it stays dense."""
    if proj_name in EXCLUDED:
        return None
    return next(chi for bound, chi in SCHEDULE if block < bound)

def leg_dims(proj_name: str) -> tuple[list[int], list[int]]:
    """
        (ket, bra) factorisations for one projection matrix. 
        nn.Linear weights are (out_dim, in_dim), so ket indexes the output side and bra the side contracted with the input.
    """
    if proj_name == "down_proj":                    # (4096, 11008)
        return DIMS_SA, DIMS_MLP
    if proj_name in ("gate_proj", "up_proj"):       # (11008, 4096)
        return DIMS_MLP, DIMS_SA
    return DIMS_SA, DIMS_SA                         # (4096, 4096)


def get_mpo_params(dims_ket: list[int], dims_bra: list[int], chi: int) -> int:
    """Returns the total number of parameters of a MPO representation of a tensor with the given ket and bra dimensions and bond dimension chi."""
    d = [a * b for a, b in zip(dims_ket, dims_bra)] # dimensions of each MPO tensor
    bonds = [1] + [min(chi, math.prod(d[: k + 1]), math.prod(d[k + 1 :])) for k in range(len(d) - 1)] + [1]
    return sum(bonds[k] * d[k] * bonds[k + 1] for k in range(len(d))) # contract the ket and bra legs of each MPO tensor with the bond legs on either side, and sum the parameter counts of all MPO tensors.


def get_compressed_and_dense_model_n_params() -> tuple[int, int]:
    """Returns the number of compressed parameters under the schedule above and the full model parameters ."""
    n_layers, compressed, dense = m.get_number_of_layers(), 0, 0
    for block in range(n_layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            ket, bra = leg_dims(name)
            full = math.prod(ket) * math.prod(bra)
            chi = chi_for(block, name)
            compressed += full if chi is None else get_mpo_params(ket, bra, chi)
            dense += full
    other = 2 * 32000 * 4096 + (2 * n_layers + 1) * 4096  # embeddings, lm_head, norms
    return compressed + other, dense + other


if __name__ == "__main__":
    kept, total = get_compressed_and_dense_model_n_params()
    print(f"schedule keeps {kept / 1e9:.2f} B of {total / 1e9:.2f} B parameters "
          f"({kept / total:.1%}, {1 - kept / total:.1%} reduction)\n", flush=True)

    out = Path(__file__).parent.parent.parent / "data" / "compressed_weights" / "schedule"
    out.mkdir(parents=True, exist_ok=True)

    for block in range(m.get_number_of_layers()):
        # merge the self-attention and MLP projection matrices of the block into one dictionary
        layers = m.get_decoder_layer(block, layer_type="self_attn") | m.get_decoder_layer(block, layer_type="mlp")

        for name, tensor in layers.items():
            chi = chi_for(block, name)
            path = out / f"layer_{block}_{name}.pt"
            if chi is None or path.exists():
                print(f"block {block:2d} {name:10} {'excluded, stays dense' if chi is None else 'already done'}", flush=True)
                continue

            ket, bra = leg_dims(name)
            mpo = c.compress_tensor(tensor, chi, ket, bra)
            save_file(c.mpo_train_to_safetensor_dict(mpo), path)
            n = sum(t.numel() for t in mpo)
            print(f"block {block:2d} {name:10} chi={chi:<5} {n:>10,} params  {n / tensor.numel():>6.1%}", flush=True)
