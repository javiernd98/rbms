import numpy as np
import torch
from torch import Tensor

from rbms.bernoulli_bernoulli_conditional.classes import BBCRBM
from rbms.bernoulli_bernoulli_conditional.implement import (
    _compute_energy_cond,
    _compute_energy_hiddens_cond,
    _compute_energy_visibles_cond,
    _compute_gradient_cond,
    _get_dynamic_biases,
    _init_chains_cond,
    _init_parameters_cond,
    _sample_hiddens_cond,
    _sample_visibles_cond,
)
from rbms.dataset.dataset_class import RBMDataset


def sample_hiddens(
    chains: dict[str, Tensor], params: BBCRBM, beta: float = 1.0, context: Tensor | None = None
) -> dict[str, Tensor]:
    """Sample the hidden layer conditionally to the visible one and the context.

    Args:
        chains (dict[str, Tensor]): The parallel chains used for sampling.
        params (BBCRBM): The parameters of the conditional RBM.
        beta (float, optional): The inverse temperature. Defaults to 1.0.
        context (Tensor | None, optional): The temporal context u. Defaults to None.

    Returns:
        dict[str, Tensor]: The updated chains with sampled hidden states.
    """
    u = chains.get("context")
    return params.sample_hiddens(chains=chains, beta=beta, context=u)


def sample_visibles(
    chains: dict[str, Tensor], params: BBCRBM, beta: float = 1.0, context: Tensor | None = None
) -> dict[str, Tensor]:
    """Sample the visible layer conditionally to the hidden one and the context.

    Args:
        chains (dict[str, Tensor]): The parallel chains used for sampling.
        params (BBCRBM): The parameters of the conditional RBM.
        beta (float, optional): The inverse temperature. Defaults to 1.0.
        context (Tensor | None, optional): The temporal context u. Defaults to None.

    Returns:
        dict[str, Tensor]: The updated chains with sampled visible states.
    """
    u = chains.get("context")
    return params.sample_visibles(chains=chains, beta=beta, context=u)


def compute_energy(
    v: Tensor,
    h: Tensor,
    params: BBCRBM,
    context: Tensor | None = None,
) -> Tensor:
    """Compute the energy of the cRBM on the visible and hidden variables given a context.

    Args:
        v (Tensor): Visible configurations.
        h (Tensor): Hidden configurations.
        params (BBCRBM): Parameters of the conditional RBM.
        context (Tensor | None, optional): The temporal context u. Defaults to None.

    Returns:
        Tensor: The computed energy.
    """
    return params.compute_energy(v=v, h=h, context=context)


def compute_energy_visibles(v: Tensor, params: BBCRBM, context: Tensor | None = None) -> Tensor:
    """Returns the marginalized energy of the model computed on the visible configurations

    Args:
        v (Tensor): Visible configurations
        context (Tensor): past visible configurations
        params (BBCRBM): Parameters of the CRBM
    Returns:
        Tensor: The computed energy.
    """
    return params.compute_energy_visibles(v=v, context=context)


def compute_energy_hiddens(h: Tensor, params: BBCRBM, context: Tensor | None = None) -> Tensor:
    """Returns the marginalized energy of the model computed on the visible configurations

    Args:
        h (Tensor): Hidden configurations.
        context (Tensor): past visible configurations
        params (BBCRBM): Parameters of the CRBM
    """
    return params.compute_energy_hiddens(h=h, context=context)


def compute_gradient(
    data: dict[str, Tensor],
    chains: dict[str, Tensor],
    params: BBCRBM,
    centered: bool = True,
) -> None:
    """Compute the gradient for each of the parameters and attach it."""
    _compute_gradient_cond(
        v_data=data["visible"],
        mh_data=data["hidden_mag"],
        u_data=data.get("context"),
        w_data=data["weights"],
        v_chain=chains["visible"],
        h_chain=chains["hidden_mag"],
        u_chain=chains.get("context"),
        w_chain=chains["weights"],
        vbias=params.vbias,
        hbias=params.hbias,
        weight_matrix=params.weight_matrix,
        A=params.A,
        B=params.B,
        centered=centered,
    )


def init_chains(
    num_samples: int,
    params: BBCRBM,
    weights: Tensor | None = None,
    start_v: Tensor | None = None,
) -> dict[str, Tensor]:
    """Initialize a Markov chain for the conditional RBM."""
    visible, hidden, mean_visible, mean_hidden = _init_chains_cond(
        num_samples=num_samples,
        weight_matrix=params.weight_matrix,
        hbias=params.hbias,
        start_v=start_v,
    )
    if weights is None:
        weights = torch.ones(visible.shape[0], device=visible.device, dtype=visible.dtype)
    return dict(
        visible=visible,
        hidden=hidden,
        visible_mag=mean_visible,
        hidden_mag=mean_hidden,
        weights=weights,
    )


def init_parameters(
    num_hiddens: int,
    dataset: RBMDataset,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
    n_past: int = 1,
) -> BBCRBM:
    """Initialize the parameters of the conditional RBM."""
    data = dataset.data
    # Convert to torch Tensor if necessary
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(dataset.data).to(device=device, dtype=dtype)
        
    vbias, hbias, weight_matrix, A, B = _init_parameters_cond(
        num_hiddens=num_hiddens,
        data=data,
        device=device,
        dtype=dtype,
        var_init=var_init,
        n_past=n_past,
    )
    return BBCRBM(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias, A=A, B=B, n_past=n_past)
