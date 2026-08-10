from llm_compression_demo import model_loading as m

d = 4096 # hidden dimension of Llama2-7b
n_decoder_blocks = 32 # number of decoder blocks in Llama2-7b
mlp_hidden_dim = 11008 # hidden dimension of the MLP in Llama2-7b

def test_model_loading():
    m.print_model_info()

    attn_layer_0 = m.get_decoder_layer(0, layer_type="self_attn")
    mlp_layer_0 = m.get_decoder_layer(0, layer_type="mlp")

    # check dimensions of the projection matrices
    assert attn_layer_0["q_proj"].shape == (d, d), "Query projection matrix shape mismatch"
    assert attn_layer_0["k_proj"].shape == (d, d), "Key projection matrix shape mismatch"
    assert attn_layer_0["v_proj"].shape == (d, d), "Value projection matrix shape mismatch"
    assert attn_layer_0["o_proj"].shape == (d, d), "Output projection matrix shape mismatch"
    assert mlp_layer_0["gate_proj"].shape == (mlp_hidden_dim, d), "Gate projection matrix shape mismatch"
    assert mlp_layer_0["up_proj"].shape == (mlp_hidden_dim, d), "Up projection matrix shape mismatch"
    assert mlp_layer_0["down_proj"].shape == (d, mlp_hidden_dim), "Down projection matrix shape mismatch"

    n_layers = m.get_number_of_layers()
    assert n_layers == n_decoder_blocks, f"Expected {n_decoder_blocks} decoder blocks, but found {n_layers}" 