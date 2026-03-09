from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from rbms.bernoulli_bernoulli_conditional.implement import (
    _compute_energy_cond,
    _compute_energy_hiddens_cond,
    _compute_energy_visibles_cond,
    _compute_gradient_cond,
    _init_chains_cond,
    _init_parameters_cond,
    _sample_hiddens_cond,
    _sample_visibles_cond,
    _get_dynamic_biases,
)
from rbms.classes import RBM
from rbms.custom_fn import check_keys_dict


class BBCRBM(RBM):
    """Parameters of the Bernoulli-Bernoulli Conditional RBM"""

    visible_type: str = "bernoulli"

    def __init__(
        self,
        weight_matrix: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        A: Tensor,
        B: Tensor,
        n_past: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the parameters of the Bernoulli-Bernoulli Conditional RBM.
        
        Args:
            weight_matrix (Tensor): The weight matrix of the RBM.
            vbias (Tensor): The visible bias of the RBM.
            hbias (Tensor): The hidden bias of the RBM.
            A (Tensor): The weight matrix connecting past visibles to present ones.
            B (Tensor): " " " " " " to present hiddens.
            device (Optional[torch.device], optional): The device for the parameters.
                Defaults to the device of `weight_matrix`.
            dtype (Optional[torch.dtype], optional): The data type for the parameters.
                Defaults to the data type of `weight_matrix`.
        """
        if device is None:
            device = weight_matrix.device
        if dtype is None:
            dtype = weight_matrix.dtype
        self.device = device
        self.dtype = dtype
        self.weight_matrix = weight_matrix.to(device=self.device, dtype=self.dtype)
        self.vbias = vbias.to(device=self.device, dtype=self.dtype)
        self.hbias = hbias.to(device=self.device, dtype=self.dtype)
        # Conditional parameters
        self.A = A.to(device=self.device, dtype=self.dtype)
        self.B = B.to(device=self.device, dtype=self.dtype)
        self.n_past = n_past
        
        self.name = "BBCRBM"
        self.flags = []

    def __add__(self, other):
        return BBCRBM(
            weight_matrix=self.weight_matrix + other.weight_matrix,
            vbias=self.vbias + other.vbias,
            hbias=self.hbias + other.hbias,
            A=self.A + other.A,
            B=self.B + other.B,
            n_past=self.n_past
        )

    def __mul__(self, other):
        return BBCRBM(
            weight_matrix=self.weight_matrix * other,
            vbias=self.vbias * other,
            hbias=self.hbias * other,
            A=self.A * other,
            B=self.B * other,
            n_past=self.n_past
        )

    def clone(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return BBCRBM(
            weight_matrix=self.weight_matrix.clone(),
            vbias=self.vbias.clone(),
            hbias=self.hbias.clone(),
            A=self.A.clone(),
            B=self.B.clone(),
            n_past=self.n_past,
            device=device,
            dtype=dtype,
        )

    def compute_energy(self, v: Tensor, h: Tensor, context: Tensor | None = None) -> Tensor:
        dyn_vbias, dyn_hbias = _get_dynamic_biases(context, self.vbias, self.hbias, self.A, self.B)
        return _compute_energy_cond(
            v=v, 
            h=h, 
            dyn_vbias=dyn_vbias, 
            dyn_hbias=dyn_hbias, 
            weight_matrix=self.weight_matrix
        )

    def compute_energy_hiddens(self, h: Tensor, context: Tensor | None = None) -> Tensor:
        dyn_vbias, dyn_hbias = _get_dynamic_biases(context, self.vbias, self.hbias, self.A, self.B)
        return _compute_energy_hiddens_cond(
            h=h, 
            dyn_vbias=dyn_vbias, 
            dyn_hbias=dyn_hbias, 
            weight_matrix=self.weight_matrix
        )

    def compute_energy_visibles(self, v: Tensor, context: Tensor | None = None) -> Tensor:
        dyn_vbias, dyn_hbias = _get_dynamic_biases(context, self.vbias, self.hbias, self.A, self.B)
        return _compute_energy_visibles_cond(
            v=v, 
            dyn_vbias=dyn_vbias, 
            dyn_hbias=dyn_hbias, 
            weight_matrix=self.weight_matrix
        )

    def compute_gradient(self, data, chains, centered=True):
        _compute_gradient_cond(
            v_data=data["visible"],
            mh_data=data["hidden_mag"],
            u_data=data.get("context"),
            w_data=data["weights"],
            v_chain=chains["visible"],
            h_chain=chains["hidden_mag"],
            u_chain=chains.get("context"),
            w_chain=chains["weights"],
            vbias=self.vbias,
            hbias=self.hbias,
            weight_matrix=self.weight_matrix,
            A=self.A,
            B=self.B,
            centered=centered,
        )

    def independent_model(self):
        return BBCRBM(
            weight_matrix=torch.zeros_like(self.weight_matrix),
            vbias=self.vbias,
            hbias=torch.zeros_like(self.hbias),
            A=torch.zeros_like(self.A),
            B=torch.zeros_like(self.B),
            n_past=self.n_past
        )

    def init_chains(self, num_samples, weights=None, start_v=None):
        # Initializing chains for a conditional model functions the same way
        # as a standard RBM, but uses the conditional initialization script.
        visible, hidden, mean_visible, mean_hidden = _init_chains_cond(
            num_samples=num_samples,
            weight_matrix=self.weight_matrix,
            hbias=self.hbias,
            start_v=start_v,
        )
        if weights is None:
            weights = torch.ones(
                visible.shape[0], device=visible.device, dtype=visible.dtype
            )
        return dict(
            visible=visible,
            hidden=hidden,
            visible_mag=mean_visible,
            hidden_mag=mean_hidden,
            weights=weights,
            # Context is usually passed batch-by-batch during sampling, 
            # so we don't necessarily initialize it here.
        )

    @staticmethod
    def init_parameters(num_hiddens, dataset, device, dtype, var_init=0.0001, n_past=1):
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
            n_past=n_past
        )
        return BBCRBM(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias, A=A, B=B, n_past=n_past)

    def named_parameters(self) -> dict[str, np.ndarray]:
        return {
            "weight_matrix": self.weight_matrix.cpu().numpy(),
            "vbias": self.vbias.cpu().numpy(),
            "hbias": self.hbias.cpu().numpy(),
            "A": self.A.cpu().numpy(),
            "B": self.B.cpu().numpy(),
            "n_past": np.asarray(self.n_past)
        }

    @property
    def num_hiddens(self):
        return self.hbias.shape[0]

    @property
    def num_visibles(self):
        return self.vbias.shape[0]

    def parameters(self) -> list[Tensor]:
        return [self.weight_matrix, self.vbias, self.hbias, self.A, self.B]

    @property
    def ref_log_z(self):
        return (
            torch.log1p(torch.exp(self.vbias)).sum() + self.num_hiddens * np.log(2)
        ).item()

    def sample_hiddens(self, chains: dict[str, Tensor], beta=1, context: Tensor | None = None) -> dict[str, Tensor]:
        dyn_vbias, dyn_hbias = _get_dynamic_biases(context, self.vbias, self.hbias, self.A, self.B)
        chains["hidden"], chains["hidden_mag"] = _sample_hiddens_cond(
            v=chains["visible"], 
            weight_matrix=self.weight_matrix, 
            dyn_hbias=dyn_hbias, 
            beta=beta,
        )
        return chains

    def sample_visibles(self, chains: dict[str, Tensor], beta=1, context: Tensor | None = None) -> dict[str, Tensor]:
        dyn_vbias, dyn_hbias = _get_dynamic_biases(context, self.vbias, self.hbias, self.A, self.B)
        chains["visible"], chains["visible_mag"] = _sample_visibles_cond(
            h=chains["hidden"], 
            weight_matrix=self.weight_matrix, 
            dyn_vbias=dyn_vbias, 
            beta=beta,
        )
        return chains

    @staticmethod
    def set_named_parameters(
        named_params: dict[str, np.ndarray],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> BBCRBM:
        names = ["vbias", "hbias", "weight_matrix", "A", "B", "n_past"]
        check_keys_dict(d=named_params, names=names)
        
        n_past_val = int(named_params.pop("n_past"))
        
        params = BBCRBM(
            weight_matrix=torch.from_numpy(named_params.pop("weight_matrix")).to(
                device=device, dtype=dtype
            ),
            vbias=torch.from_numpy(named_params.pop("vbias")).to(
                device=device, dtype=dtype
            ),
            hbias=torch.from_numpy(named_params.pop("hbias")).to(
                device=device, dtype=dtype
            ),
            A=torch.from_numpy(named_params.pop("A")).to(
                device=device, dtype=dtype
            ),
            B=torch.from_numpy(named_params.pop("B")).to(
                device=device, dtype=dtype
            ),
            n_past=n_past_val
        )
        
        if len(named_params.keys()) > 0:
            raise ValueError(
                f"Too many keys in params dictionary. Remaining keys: {named_params.keys()}"
            )
        return params

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.weight_matrix = self.weight_matrix.to(device=self.device, dtype=self.dtype)
        self.vbias = self.vbias.to(device=self.device, dtype=self.dtype)
        self.hbias = self.hbias.to(device=self.device, dtype=self.dtype)
        self.A = self.A.to(device=self.device, dtype=self.dtype)
        self.B = self.B.to(device=self.device, dtype=self.dtype)
        return self

    def get_metrics(self, metrics):
        return metrics

    def post_grad_update(self):
        pass

    def pre_grad_update(self):
        pass