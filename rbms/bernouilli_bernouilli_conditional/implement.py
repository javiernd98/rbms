import torch
from torch import Tensor

@torch.jit.script
def _get_dynamic_biases(vbias: Tensor, hbias: Tensor, A: Tensor, B: Tensor, context: Tensor):
    # dynamic_vbias = b + uA; dynamic_hbias = c + uB
    dyn_vbias = vbias + (context @ A)
    dyn_hbias = hbias + (context @ B)
    return dyn_vbias, dyn_hbias

@torch.jit.script
def _sample_hiddens_cond(v: Tensor, weight_matrix: Tensor, dyn_hbias: Tensor, beta: float = 1.0):
    mh = torch.sigmoid(beta * (dyn_hbias + (v @ weight_matrix)))
    h = torch.bernoulli(mh)
    return h, mh

@torch.jit.script
def _sample_visibles_cond(h: Tensor, weight_matrix: Tensor, dyn_vbias: Tensor, beta: float = 1.0):
    mv = torch.sigmoid(beta * (dyn_vbias + (h @ weight_matrix.T)))
    v = torch.bernoulli(mv)
    return v, mv

def _compute_gradient_cond(
    v_data: Tensor, mh_data: Tensor, u_data: Tensor,
    v_chain: Tensor, h_chain: Tensor, u_chain: Tensor,
    weight_matrix: Tensor, vbias: Tensor, hbias: Tensor, A: Tensor, B: Tensor,
    centered: bool = True
):
    # Implementation of centered gradients for A and B follows the same 
    # logic as BBRBM, tracking the mean of the context vector u.
    # ... (centering logic for W, vbias, hbias, A, B) ...
    pass

# @torch.jit.script
def _compute_gradient_cond(
    v_data: Tensor,
    mh_data: Tensor,
    u_data: Tensor,
    w_data: Tensor,
    v_chain: Tensor,
    h_chain: Tensor,
    u_chain: Tensor,
    w_chain: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    A: Tensor,
    B: Tensor,
    weight_matrix: Tensor,
    centered: bool = True,
) -> None:
    w_data = w_data.view(-1, 1)
    w_chain = w_chain.view(-1, 1)
    # Turn the weights of the chains into normalized weights
    chain_weights = w_chain / w_chain.sum()
    w_data_norm = w_data.sum()

    # Averages over data and generated samples
    v_data_mean = (v_data * w_data).sum(0) / w_data_norm
    torch.clamp_(v_data_mean, min=1e-7, max=(1.0 - 1e-7))
    h_data_mean = (mh_data * w_data).sum(0) / w_data_norm
    v_gen_mean = (v_chain * chain_weights).sum(0)
    torch.clamp_(v_gen_mean, min=1e-7, max=(1.0 - 1e-7))
    h_gen_mean = (h_chain * chain_weights).sum(0)

    if centered:
        # Centered variables
        v_data_centered = v_data - v_data_mean
        h_data_centered = mh_data - h_data_mean
        v_gen_centered = v_chain - v_data_mean
        h_gen_centered = h_chain - h_data_mean

        # Gradient
        grad_weight_matrix = (
            (v_data_centered * w_data).T @ h_data_centered
        ) / w_data_norm - ((v_gen_centered * chain_weights).T @ h_gen_centered)
        grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
        grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
    else:
        # Gradient
        grad_weight_matrix = ((v_data * w_data).T @ mh_data) / w_data_norm - (
            (v_chain * chain_weights).T @ h_chain
        )
        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean

    # Attach to the parameters

    weight_matrix.grad = grad_weight_matrix
    vbias.grad = grad_vbias
    hbias.grad = grad_hbias