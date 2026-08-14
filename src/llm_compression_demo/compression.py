import math, torch
from llm_compression_demo import model_loading as m

LOCAL_DIM = 8 # best compromise between number of tensors and contraction cost. 

def reshape_SA_layer(w: torch.Tensor) -> torch.Tensor:
    """
        The SA weight matrices have dimension 4096 x 4096.
        The goal of the reshaping is, to minimize the number of tensors so truncation errors do not accumulate but also look at the contraction cost which is O(d*chi^2)
    """

    assert (math.log(w.shape[0], LOCAL_DIM)).is_integer(), f"{w.shape[0]} != {LOCAL_DIM}^N for any integer N. Please choose a different LOCAL_DIM."
    N = int(math.log(w.shape[0], LOCAL_DIM))

    return w.reshape(*[LOCAL_DIM for _ in range(0, 2*N)])

def reshape_MLP_layer(w: torch.Tensor) -> torch.Tensor:
    """
        The MLP weight matrices have dimension 11008 x 4096.
        The goal of the reshaping is, to minimize the number of tensors so truncation errors do not accumulate but also look at the contraction cost which is O(d*chi^2)

        We use this layout:

        ```
            8|   8|   4|  43|
             O -- O -- O -- O
            8|   8|   8|   8|

        ```
    """
    assert LOCAL_DIM == 8, "The MLP layer reshaping is hardcoded for LOCAL_DIM=8. Please change the code if you want to use a different LOCAL_DIM."
    return w.reshape(8, 8, 4, 43, 8, 8, 8, 8)

def interleave_ket_bra(t: torch.Tensor) -> torch.Tensor:
    """
        Interleaves the ket and bra legs of a tensor t with shape [d1, d2, ..., dN, d1', d2', ..., dN'] into a tensor with shape [d1, d1', d2, d2', ..., dN, dN'].
    """
    assert t.ndim % 2 == 0, f"t must have an even number of dimensions, but has {t.ndim} dimensions."
    N = t.ndim // 2
    return t.permute(*[i for k in range(N) for i in (k, k+N)])

def reverse_interleave_ket_bra(t: torch.Tensor) -> torch.Tensor:
    """
        Reverses the interleaving of the ket and bra legs of a tensor t with shape [d1, d1', d2, d2', ..., dN, dN'] into a tensor with shape [d1, d2, ..., dN, d1', d2', ..., dN'].
    """
    assert t.ndim % 2 == 0, f"t must have an even number of dimensions, but has {t.ndim} dimensions."
    N = t.ndim // 2
    return t.permute(*range(0, t.ndim, 2), *range(1, t.ndim, 2))

def compress_tensor(t: torch.Tensor, chi: int, dims_ket: list, dims_bra: list) -> list[torch.Tensor]:
    """
        Creates a MPO representation of the tensor `t` with bond dimension `chi`.

        ### Arguments
        - `t: torch.Tensor`
            The tensor to be compressed. Must have shape (prod(dims_ket), prod(dims_bra)).
        - `chi: int`
            The maximum bond dimension of the MPO.
        - `dims_ket: list`
            The dimensions of the ket legs of the tensor.
        - `dims_bra: list`
            The dimensions of the bra legs of the tensor.
    """
    N = len(dims_ket)
    assert len(dims_bra) == N, "one bra leg per ket leg"
    assert tuple(t.shape) == (math.prod(dims_ket), math.prod(dims_bra)), f"{tuple(t.shape)} != {(math.prod(dims_ket), math.prod(dims_bra))}"

    t = interleave_ket_bra(t.reshape(*dims_ket, *dims_bra))

    # fixed chi at every bond
    mpo_train, left = [], 1
    for k in range(N - 1): # sweep from left to right
        u, s, v = torch.linalg.svd(t.reshape(left * dims_ket[k] * dims_bra[k], -1), full_matrices=False)
        r = min(chi, s.shape[0]) # truncate to chi or the exact rank, whichever is smaller
        mpo_train.append(u[:, :r].contiguous().reshape(left, dims_ket[k], dims_bra[k], r)) # add matrix u to the mpo_train

        t = s[:r, None] * v[:r] # next tensor in the chain
        left = r

    # last tensor in the chain is the leftover t = s[:r, None] * v[:r]
    mpo_train.append(t.reshape(left, dims_ket[-1], dims_bra[-1], 1))
    return mpo_train

def mpo_train_to_safetensor_dict(mpo_train: list[torch.Tensor], dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    """
        Converts a MPO representation of a tensor into a dictionary of tensors that can be saved as a safetensor file.
    """
    return {f"mpo_{k}": t.to(dtype=dtype) for k, t in enumerate(mpo_train)}

if __name__ == "__main__":
    # At full chi the MPO is exact. This is the check that catches a wrong ket/bra interleave:
    # a bad permute still round-trips shapes fine, but the numbers come out transposed.
    torch.manual_seed(0)
    dims_ket, dims_bra = [8,8,8,8], [8,8,8,8]
    w = torch.randn(4096, 4096, dtype=torch.float64)
    mpo_train = compress_tensor(w, 2, dims_ket, dims_bra)

    print(mpo_train[0])
    print(mpo_train[0].shape)
