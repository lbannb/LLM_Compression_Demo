import torch
from torch import nn
from llm_compression_demo.build_compressed_model import MPOLinear
from llm_compression_demo import model_loading as m

def test_mpo_linear():
    """
        Test the MPOLinear module by comparing its output with the output of a standard linear layer.
    """
    torch.manual_seed(0)

    w = m.get_decoder_layer(0)["q_proj"]
    mpo = MPOLinear.from_file("data/compressed_weights/chi=20/layer_0_q_proj.pt",
                              dtype=w.dtype, device=str(w.device))
    print(mpo)

    x = torch.randn(3, 7, mpo.in_dim, dtype=w.dtype, device=w.device) # random input tensor with batch size 3 and sequence length 7
    ours, target = mpo(x), nn.functional.linear(x, w)
    err = ((ours - target).norm() / target.norm()).item() # compute the relative error between the outputs of the MPOLinear module and the standard linear layer

    assert ours.shape == target.shape, f"{tuple(ours.shape)} != {tuple(target.shape)}"
    print(f"shape {tuple(ours.shape)}  relative error vs dense weight: {err:.4f}")
    assert err < 1.0, "output no better than predicting zero -- check the ket/bra leg ordering"

# def test_sentence_completion():

if __name__ == "__main__":
    test_mpo_linear()
    # test_sentence_completion()
    
