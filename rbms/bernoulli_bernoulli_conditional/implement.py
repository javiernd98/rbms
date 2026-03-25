import torch
from torch import Tensor

@torch.jit.script
def _get_dynamic_biases(
    u: Tensor | None, vbias: Tensor, hbias: Tensor, A: Tensor, B: Tensor
) -> tuple[Tensor, Tensor]:
    """Computes the dynamic biases by adding the context-dependent terms."""
    if u is not None:
        return vbias + (u @ A), hbias + (u @ B)
    return vbias, hbias

@torch.jit.script
def _sample_hiddens_cond(
    v: Tensor, weight_matrix: Tensor, dyn_hbias: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    mh = torch.sigmoid(beta * (dyn_hbias + (v @ weight_matrix)))
    h = torch.bernoulli(mh)
    return h, mh


@torch.jit.script
def _sample_visibles_cond(
    h: Tensor, weight_matrix: Tensor, dyn_vbias: Tensor, beta: float = 1.0
) -> tuple[Tensor, Tensor]:
    mv = torch.sigmoid(beta * (dyn_vbias + (h @ weight_matrix.T)))
    v = torch.bernoulli(mv)
    return v, mv


@torch.jit.script
def _compute_energy_cond(
    v: Tensor, h: Tensor, dyn_vbias: Tensor, dyn_hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    fields = torch.multiply(v, dyn_vbias).sum(1) + torch.multiply(h, dyn_hbias).sum(1)
    interaction = torch.multiply(
        v, torch.tensordot(h, weight_matrix, dims=[[1], [1]])
    ).sum(1)

    return -fields - interaction


@torch.jit.script
def _compute_energy_visibles_cond(
    v: Tensor, dyn_vbias: Tensor, dyn_hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    field = torch.multiply(v, dyn_vbias).sum(1)
    exponent = dyn_hbias + (v @ weight_matrix)
    log_term = torch.where(exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent)    
    return -field - log_term.sum(1)


@torch.jit.script
def _compute_energy_hiddens_cond(
    h: Tensor, dyn_vbias: Tensor, dyn_hbias: Tensor, weight_matrix: Tensor
) -> Tensor:
    field = torch.multiply(h, dyn_hbias).sum(1)
    exponent = dyn_vbias + (h @ weight_matrix.T)
    log_term = torch.where(exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent)    
    return -field - log_term.sum(1)


# @torch.jit.script
def _compute_gradient_cond(
    v_data: Tensor,
    mh_data: Tensor,
    u_data: Tensor | None,
    w_data: Tensor,
    v_chain: Tensor,
    h_chain: Tensor,
    u_chain: Tensor | None,
    w_chain: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    weight_matrix: Tensor,
    A: Tensor,
    B: Tensor,
    centered: bool = True,
) -> None:
    if u_data is None or u_chain is None:
        raise ValueError("Context 'u' cannot be None for conditional RBM gradients.")

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

    # Note: u is identical for data and chains in standard CD/PCD for cRBMs
    u_data_mean = (u_data * w_data).sum(0) / w_data_norm

    if centered:
        # Centered variables
        v_data_centered = v_data - v_data_mean
        h_data_centered = mh_data - h_data_mean
        u_data_centered = u_data - u_data_mean       
        v_gen_centered = v_chain - v_data_mean
        h_gen_centered = h_chain - h_data_mean
        u_gen_centered = u_chain - u_data_mean

        # Gradients
        grad_weight_matrix = (
            (v_data_centered * w_data).T @ h_data_centered
        ) / w_data_norm - ((v_gen_centered * chain_weights).T @ h_gen_centered)

        grad_A = (
            (u_data_centered * w_data).T @ v_data_centered
        ) / w_data_norm - ((u_gen_centered * chain_weights).T @ v_gen_centered)

        grad_B = (
            (u_data_centered * w_data).T @ h_data_centered
        ) / w_data_norm - ((u_gen_centered * chain_weights).T @ h_gen_centered)
        # Biases must account for the centering offsets of W, A, and B
        grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean) - (u_data_mean @ grad_A)
        grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix) - (u_data_mean @ grad_B)

    else:
        # Uncentered Gradient
        grad_weight_matrix = ((v_data * w_data).T @ mh_data) / w_data_norm - (
            (v_chain * chain_weights).T @ h_chain
        )
        grad_A = ((u_data * w_data).T @ v_data) / w_data_norm - (
            (u_chain * chain_weights).T @ v_chain
        )
        grad_B = ((u_data * w_data).T @ mh_data) / w_data_norm - (
            (u_chain * chain_weights).T @ h_chain
        )
        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean

    # Attach to the parameters

    weight_matrix.grad = grad_weight_matrix
    vbias.grad = grad_vbias
    hbias.grad = grad_hbias
    A.grad = grad_A
    B.grad = grad_B

def _init_chains_cond(
    num_samples: int,
    weight_matrix: Tensor,
    hbias: Tensor,
    start_v: Tensor | None = None,
):
    num_visibles, _ = weight_matrix.shape
    device = weight_matrix.device
    dtype = weight_matrix.dtype

    if num_samples <= 0:
        if start_v is not None:
            num_samples = start_v.shape[0]
        else:
            raise ValueError(f"Got negative num_samples arg: {num_samples}")

    if start_v is None:
        mv = torch.ones(size=(num_samples, num_visibles), device=device, dtype=dtype) / 2
        v = torch.bernoulli(mv)
    else:
        mv = torch.ones_like(start_v, device=device, dtype=dtype) / 2
        v = start_v.to(device=device, dtype=dtype)

    # Para arrancar la cadena por primera vez, asumimos que no hay contexto (dyn_hbias = hbias)
    h, mh = _sample_hiddens_cond(v=v, weight_matrix=weight_matrix, dyn_hbias=hbias)
    return v, h, mv, mh

def _init_parameters_cond(
    num_hiddens: int,
    data: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
    n_past: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _, num_visibles = data.shape
    history_size = num_visibles * n_past
    eps = 1e-4
    weight_matrix = (
        torch.randn(size=(num_visibles, num_hiddens), device=device, dtype=dtype)
        * var_init
    )
    frequencies = data.mean(0)
    frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
    vbias = (torch.log(frequencies) - torch.log(1.0 - frequencies)).to(
        device=device, dtype=dtype
    )
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    # Initialize conditional RBM parameters
    A = (
        torch.randn(size=(history_size, num_visibles), device=device, dtype=dtype)
        * var_init
    )
    B = (
        torch.randn(size=(history_size, num_hiddens), device=device, dtype=dtype)
         * var_init
    )
    return vbias, hbias, weight_matrix, A, B