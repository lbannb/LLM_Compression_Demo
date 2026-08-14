import torch

from llm_compression_demo import model_loading as m
from llm_compression_demo import compression as c

def test_reconstruction():
    """
        Test that the reconstruction of a tensor from its MPO representation is the original tensor when no singular values are truncated.
    """

    torch.manual_seed(0)

    # create a random tensor of shape (d,d)
    d = 216
    w = torch.randn(d, d, dtype=torch.float64)
    dims_ket = [6,6,6]
    dims_bra = [6,6,6]
    chi = min(dims_ket) * min(dims_bra)  # no truncation
    chi = 200

    w_interleave = c.interleave_ket_bra(w)
    assert torch.equal(w, c.reverse_interleave_ket_bra(w_interleave)), "Interleaving and reverse interleaving failed: the original tensor is not the same as the reconstructed tensor."

    mpo_train = c.compress_tensor(w, chi, dims_ket, dims_bra)
    
    # reconstruct the tensor from its MPO representation
    w_reconstructed = mpo_train[0]
    for k in range(1, len(mpo_train)):
        w_reconstructed = torch.tensordot(w_reconstructed, mpo_train[k], dims=1) # contract the last leg of w_reconstructed with the first leg of mpo_train[k]

    # remove boundary dimensions
    w_reconstructed = w_reconstructed.squeeze(0).squeeze(-1)
    # reshape the reconstructed tensor to the original shape
    w_reconstructed = c.reverse_interleave_ket_bra(w_reconstructed).reshape(d, d)

    # check that the reconstruction is close to the original tensor
    assert torch.allclose(w, w_reconstructed, atol=1e-15), "Reconstruction failed: the reconstructed tensor is not the same as the original."

if __name__ == "__main__":
    test_reconstruction()