import math, string, torch

from torch import nn
from safetensors import safe_open


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
        out = torch.einsum(self.equation, x.reshape(-1, *self.dims_bra), *self.mpo_train)
        return out.reshape(*x.shape[:-1], self.out_dim)

    def extra_repr(self) -> str:
        """
            Returns a string representation of the module, including the input and output dimensions and the bond dimensions
        """
        bonds = [c.shape[0] for c in self.mpo_train] + [1]
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, bonds={bonds}"
