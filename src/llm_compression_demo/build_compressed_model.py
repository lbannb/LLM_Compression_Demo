import math, string, sys, torch

from pathlib import Path
from typing import cast

from torch import nn
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from accelerate import init_empty_weights
from transformers import AutoConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llm_compression_demo import model_loading as m
from llm_compression_demo.compression import reverse_interleave_ket_bra


class MPOLinear(nn.Module):
    """
        Replacement for the nn.Linear layers of the Llama decoder blocks, with the weight
        matrix stored as an MPO instead of a dense matrix.

        Honours the nn.Linear contract, (..., in_dim) -> (..., out_dim), so HuggingFace's
        LlamaAttention / LlamaMLP keep working unchanged: the head reshape, RoPE, the causal mask and
        the softmax all live downstream of this module. Llama uses bias=False throughout, so there is
        no bias to carry.

        The leg dimensions are read back off the mpo shapes (chi_left, ket_k, bra_k, chi_right), so
        the files need no metadata.
    """

    # Token count above which forward() contracts to a dense matrix instead of streaming the
    # tensors. Below it the einsum wins (one token needs ~400k elements, the matrix needs 17M).
    dense_above = 32

    def __init__(self, mpo_train: list[torch.Tensor]):
        super().__init__() # initialize the nn.Module

        self.dims_ket = [c.shape[1] for c in mpo_train]
        self.dims_bra = [c.shape[2] for c in mpo_train]
        self.out_dim = math.prod(self.dims_ket)
        self.in_dim = math.prod(self.dims_bra)

        # nn.ParameterList is a container for nn.Parameter objects, which are tensors that are registered as parameters of the module and will be updated during training. 
        # The requires_grad=False argument means that the parameters will not be updated during training. In the healing process we flip this to True.
        self.mpo_train = nn.ParameterList(nn.Parameter(c, requires_grad=False) for c in mpo_train)

        """ 
            Builds the einsum equation once, for forward() to reuse.

            For the einsum we use latters of the alphabet for the ket and bra legs and for the bonds. 
            The first operand is the input, which has a batch dimension 'a' and one bra leg per site:
            
                --
               |  |-b-
            -a-|  |-c-
               |  |-d-
               |  |-e-
                --
            
            Here `a` is the flattened batch `B*T` where `B` is the batch size and `T` is the sequence length.

            The following mpo tensors then carry a bra and ket layer as well as the bonds:


                     |b     |c     |d     |e
                    --     --     --     --
                -j-|  |-k-|  |-l-|  |-m-|  |-n-
                    --     --     --     --
                     |f     |g     |h     |i

            The two boundary bonds `j` and `n` are size 1, so leaving them out of the output makes einsum sum them away.
        """
        N, letters = len(mpo_train), iter(string.ascii_letters)
        batch = next(letters) # a
        bra = [next(letters) for _ in range(N)] # b, c, d, e
        ket = [next(letters) for _ in range(N)] # f, g, h, i
        bond = [next(letters) for _ in range(N + 1)] # j, k, l, m, n

        operands = [batch + "".join(bra)] + [bond[k] + ket[k] + bra[k] + bond[k + 1] for k in range(N)]
        self.equation = ",".join(operands) + "->" + batch + "".join(ket) # abcde,jfbk,kgcl,lhdm,mien->afghi

    @classmethod
    def from_file(cls, path: str, dtype: torch.dtype = torch.float16, device: str = "cuda") -> "MPOLinear":
        """
            Loads the mpo_train written by the compression step.
        """
        with safe_open(path, framework="pt") as f:
            keys = sorted(f.keys(), key=lambda k: int(k.rsplit("_", 1)[1]))
            return cls([f.get_tensor(k).to(device=device, dtype=dtype) for k in keys])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass of the MPOLinear module.
            Absorbs the mpo_train one at a time over the whole batch.

            ### Arguments
            - `x: torch.Tensor`
                The input tensor with shape (..., in_dim).

            ### Returns
            - `torch.Tensor`
                The output tensor with shape (..., out_dim).
        """
        if x.numel() // self.in_dim >= self.dense_above:
            return nn.functional.linear(x, self.to_dense())

        out = torch.einsum(self.equation, x.reshape(-1, *self.dims_bra), *self.mpo_train)
        return out.reshape(*x.shape[:-1], self.out_dim)

    def to_dense(self) -> torch.Tensor:
        """
            Contracts the mpo tensors back into the (out_dim, in_dim) matrix.

            Mathematically identical to the einsum path, but the cost is fixed by the matrix rather
            than by the token count: streaming the tensors over m tokens builds intermediates of
            ~m * 400k elements, while this builds out_dim * in_dim once and then hands cuBLAS a
            plain dense matrix multiplication. 
            Above a few dozen tokens that is both smaller and faster, and during
            training it is the difference between fitting on my 2080 Ti card and not.
        """
        r = self.mpo_train[0]
        for t in self.mpo_train[1:]: # contract the mpo tensors one at a time
            r = torch.tensordot(r, t, dims=1)

        r = r.squeeze(-1).squeeze(0)  # remove the boundary bonds of size 1
        return reverse_interleave_ket_bra(r).reshape(self.out_dim, self.in_dim) # undo the interleave and then flatten to (out_dim, in_dim)

    def extra_repr(self) -> str:
        """
            Returns a string representation of the module, including the input and output dimensions and the bond dimensions
        """
        bonds = [c.shape[0] for c in self.mpo_train] + [1]
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, bonds={bonds}"


def build_compressed_model(chi_dir: str, dtype: torch.dtype = torch.float16, device: str = "cuda") -> LlamaForCausalLM:
    """
        Builds Llama-2-7b with all 7 projections per decoder block replaced by their MPO factorisation.

        The skeleton is created on the meta device, so the 13.5 GB of dense weights are never
        allocated anywhere: the projections are overwritten with MPOLinear before anything is filled,
        and only the remainder (embeddings, lm_head, the norms -- 525 MB) is read off the checkpoint.
    """

    SA_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
    MLP_PROJ = ("gate_proj", "up_proj", "down_proj")

    # init_empty_weights() context manager creates a model with all parameters on the meta device, which is a special device that does not allocate any memory for the parameters. 
    # This allows us to create a model with a large number of parameters without running out of memory.
    with init_empty_weights():
        model = LlamaForCausalLM(AutoConfig.from_pretrained(m.REPO))

    # nn.ModuleList yields a bare Module to a type checker, which loses self_attn and mlp.
    # A matrix is compressed iff a file exists for it. The sensitivity schedule leaves down_proj
    # dense in every block.
    swapped = set()
    for i, layer in enumerate(cast(list[LlamaDecoderLayer], model.model.layers)):
        for parent, names, kind in ((layer.self_attn, SA_PROJ, "self_attn"), (layer.mlp, MLP_PROJ, "mlp")):
            for name in names:
                path = Path(chi_dir) / f"layer_{i}_{name}.pt"
                if path.exists():
                    setattr(parent, name, MPOLinear.from_file(str(path), dtype, device)) # replace the nn.Linear with the MPOLinear
                    swapped.add(f"model.layers.{i}.{kind}.{name}.weight") # keep track of which weights have been replaced with MPOLinear

    # everything that did not get an MPO replacement
    dense = {}
    for index_value in sorted(set(m.INDEX.values())):
        with safe_open(hf_hub_download(m.REPO, index_value), framework="pt", device=device) as f:
            dense.update({k: f.get_tensor(k).to(dtype) for k in f.keys() if k not in swapped}) # load the dense weights that were not replaced with MPOLinear

    # assign=True replaces the meta tensors
    model.load_state_dict(dense, strict=False, assign=True)

    unfilled = [n for n, p in model.named_parameters() if p.is_meta] # check if there are any parameters that are still on the meta device
    assert not unfilled, f"still on the meta device: {unfilled[:5]}"
    return model.eval()


if __name__ == "__main__":
    from transformers import AutoTokenizer

    PROMPT = "The main capital of germany is"
    DIR = sys.argv[1] if len(sys.argv) > 1 else "data/compressed_weights/healed"

    model = build_compressed_model(DIR)
    print(f"parameters   : {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M")
    print(f"gpu allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    tok = AutoTokenizer.from_pretrained(m.REPO)
    ids = tok(PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=20, do_sample=False)  # type: ignore[attr-defined]

    print(f"\nprompt: {PROMPT!r}")
    print(f"output: {tok.decode(out[0], skip_special_tokens=True)!r}")
