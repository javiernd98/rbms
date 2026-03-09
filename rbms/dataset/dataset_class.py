from __future__ import annotations
import gzip
import textwrap
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

from rbms.dataset.utils import convert_data


class RBMDataset(Dataset):
    """A dataset class for RBM training and evaluation."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        names: np.ndarray,
        dataset_name: str,
        variable_type: str,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # names should stay as a np array as its dtype is object
        self.names = names
        self.dataset_name = dataset_name
        self.device = device
        self.dtype = dtype
        self.variable_type: str = variable_type
        self.data = torch.from_numpy(data).to(device=self.device, dtype=self.dtype)
        # Weights should have shape n_visibles
        self.weights = (
            torch.from_numpy(weights).view(-1).to(device=self.device, dtype=self.dtype)
        )
        # Labels are int
        self.labels = torch.from_numpy(labels).to(device=self.device, dtype=torch.int32)

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.data.shape[0]

    def __getitem__(self, index: int) -> dict[str, Union[np.ndarray, torch.Tensor]]:
        """Get a sample from the dataset.

        Args:
            index (int): The index of the sample.

        Returns:
            Dict[str, Union[np.ndarray, torch.Tensor]]: A dictionary containing the sample data, labels, weights, and names.
        """
        return {
            "data": self.data[index],
            "labels": self.labels[index],
            "weights": self.weights[index],
            "names": self.names[index],
        }

    def __str__(self) -> str:
        """Get a string representation of the dataset.

        Returns:
            str: The string representation of the dataset.
        """
        return textwrap.dedent(
            f"""
        Dataset: {self.dataset_name}
        Variable type: {self.variable_type}
        Number of samples: {self.data.shape[0]}
        Number of features: {self.data.shape[1]}
        """
        )

    def get_num_visibles(self) -> int:
        """Get the number of visible units.

        Returns:
            int: The number of visible units.
        """
        return self.data.shape[1]

    def get_num_states(self) -> int:
        """Get the number of states.

        Returns:
            int: The number of states.
        """
        return int(self.data.max() + 1)

    def get_effective_size(self) -> int:
        """Get the effective size of the dataset.

        Returns:
            int: The effective size of the dataset.
        """
        return int(self.weights.sum())

    def get_gzip_entropy(self, mean_size: int = 50, num_samples: int = 100):
        """Compute the gzip entropy of the dataset.

        Args:
            mean_size (int, optional): The number of samples to average over. Defaults to 50.
            num_samples (int, optional): The number of samples to use for each entropy calculation. Defaults to 100.

        Returns:
            float: The computed gzip entropy.
        """

        pbar = tqdm(range(mean_size))
        pbar.set_description("Compute entropy gzip")
        en = np.zeros(mean_size)
        for i in pbar:
            en[i] = len(
                gzip.compress(
                    (self.data[torch.randperm(self.data.shape[0])[:num_samples]])
                    .cpu()
                    .numpy()
                    .astype(int)
                )
            )
        return np.mean(en)

    def match_model_variable_type(self, visible_type: str):
        self.data = convert_data[self.variable_type][visible_type](self.data)
        if self.variable_type != visible_type:
            print(f"Converting from '{self.variable_type}' to '{visible_type}'")
            print(self.data)
        self.variable_type = visible_type

    def astype(self, target_variable_type: str):
        return convert_data[self.variable_type][target_variable_type](self.data)

    def split_train_test(
        self,
        rng: np.random.Generator,
        train_size: float,
        test_size: float | None = None,
    ) -> tuple[RBMDataset, RBMDataset]:
        num_samples = self.data.shape[0]
        if test_size is None:
            test_size = 1.0 - train_size

        # Shuffle dataset
        permutation_index = rng.permutation(num_samples)
        train_size = int(train_size * num_samples)
        test_size = int(test_size * num_samples)

        train_dataset = RBMDataset(
            data=self.data[permutation_index[:train_size]].cpu().numpy(),
            labels=self.labels[permutation_index[:train_size]].cpu().numpy(),
            weights=self.weights[permutation_index[:train_size]].cpu().numpy(),
            names=self.names[permutation_index[:train_size]],
            dataset_name=self.dataset_name,
            variable_type=self.variable_type,
            device=self.device,
            dtype=self.dtype,
        )
        test_dataset = None
        if test_size > 0:
            test_dataset = RBMDataset(
                data=self.data[permutation_index[train_size : train_size + test_size]]
                .cpu()
                .numpy(),
                labels=self.labels[permutation_index[train_size : train_size + test_size]]
                .cpu()
                .numpy(),
                weights=self.weights[
                    permutation_index[train_size : train_size + test_size]
                ]
                .cpu()
                .numpy(),
                names=self.names[permutation_index[train_size : train_size + test_size]],
                dataset_name=self.dataset_name,
                variable_type=self.variable_type,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError("Could not split in train test")
        return train_dataset, test_dataset

    def batch(self, batch_size: int) -> dict[str, Tensor]:
        rand_idx = torch.randperm(len(self))[:batch_size]
        sampled_batch: dict[str, Tensor] = {
            "data": self.data[rand_idx],
            "weights": self.weights[rand_idx],
            "labels": self.labels[rand_idx],
        }
        # sampled_batch = self[rand_idx[:batch_size]]
        match self.variable_type:
            case "bernoulli":
                sampled_batch["data"] = torch.bernoulli(sampled_batch["data"])
            case _:
                pass
        return sampled_batch


class CRBMDataset(RBMDataset):
    """A dataset class for Conditional RBM training and evaluation."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        names: np.ndarray,
        dataset_name: str,
        variable_type: str,
        n_past: int = 1,  # size of the temporal context
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Llamamos al init de la clase padre (RBMDataset) para que inicialice los tensores
        super().__init__(
            data=data,
            labels=labels,
            weights=weights,
            names=names,
            dataset_name=dataset_name,
            variable_type=variable_type,
            device=device,
            dtype=dtype,
        )
        self.n_past = n_past

    def __len__(self) -> int:
        """Get the number of samples in the dataset.
        En una CRBM, perdemos los primeros 'n_past' instantes porque no tienen historia.
        """
        return max(0, self.data.shape[0] - self.n_past)

    def __getitem__(self, index: int) -> dict[str, Union[np.ndarray, torch.Tensor]]:
        # El target 'v' está desplazado n_past posiciones
        idx = index + self.n_past
        
        # El contexto 'u' son los n_past pasos anteriores concatenados (flattened)
        u = self.data[index:idx].flatten()
        
        return {
            "data": self.data[idx],
            "context": u,
            "labels": self.labels[idx] if self.labels is not None else -1,
            "weights": self.weights[idx] if self.weights is not None else 1.0,
            "names": self.names[idx] if self.names is not None else "",
        }

    def batch(self, batch_size: int) -> dict[str, Tensor]:
        valid_len = len(self)
        if valid_len == 0:
            raise ValueError(f"El dataset es demasiado corto para n_past={self.n_past}.")
            
        # Elegimos instantes aleatorios pero respetando la ventana de contexto
        rand_idx = torch.randperm(valid_len)[:batch_size]
        
        v_batch, u_batch, w_batch, labels_batch = [], [], [], []
        
        for i in rand_idx:
            idx = i + self.n_past
            v_batch.append(self.data[idx])
            u_batch.append(self.data[i:idx].flatten())
            w_batch.append(self.weights[idx])
            if self.labels is not None:
                labels_batch.append(self.labels[idx])
                
        sampled_batch = {
            "data": torch.stack(v_batch),
            "context": torch.stack(u_batch),
            "weights": torch.stack(w_batch),
        }
        
        if self.labels is not None and len(labels_batch) > 0:
            sampled_batch["labels"] = torch.stack(labels_batch)
            
        if self.variable_type == "bernoulli":
            sampled_batch["data"] = torch.bernoulli(sampled_batch["data"])
            
        return sampled_batch

    def split_train_test(
        self,
        rng: np.random.Generator,
        train_size: float,
        test_size: float | None = None,
    ) -> tuple['CRBMDataset', 'CRBMDataset']:
        """
        We modify this function so it doesnt shuffle the dataset breaking the temporal otder.
        """
        num_samples = self.data.shape[0]
        if test_size is None:
            test_size = 1.0 - train_size

        train_len = int(train_size * num_samples)
        test_len = int(test_size * num_samples)

        # Hacemos split sin barajar (secuencial)
        train_dataset = CRBMDataset(
            data=self.data[:train_len].cpu().numpy(),
            labels=self.labels[:train_len].cpu().numpy() if self.labels is not None else np.zeros(0),
            weights=self.weights[:train_len].cpu().numpy(),
            names=self.names[:train_len],
            dataset_name=self.dataset_name,
            variable_type=self.variable_type,
            n_past=self.n_past,
            device=self.device,
            dtype=self.dtype,
        )
        
        test_dataset = None
        if test_size > 0:
            test_dataset = CRBMDataset(
                data=self.data[train_len : train_len + test_len].cpu().numpy(),
                labels=self.labels[train_len : train_len + test_len].cpu().numpy() if self.labels is not None else np.zeros(0),
                weights=self.weights[train_len : train_len + test_len].cpu().numpy(),
                names=self.names[train_len : train_len + test_len],
                dataset_name=self.dataset_name,
                variable_type=self.variable_type,
                n_past=self.n_past,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError("Could not split in train test")
            
        return train_dataset, test_dataset