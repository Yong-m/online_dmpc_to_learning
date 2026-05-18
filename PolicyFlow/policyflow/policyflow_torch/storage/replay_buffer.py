from typing import List, Optional, Tuple, Union
import csv
import datetime
import functools
import operator
import os
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, RandomSampler


class ReplayBuffer:
    def __init__(
        self,
        memory_size: int,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
    ) -> None:
        """Base class representing a memory with circular buffers

        Buffers are torch tensors with shape (memory size, number of environments, data size).
        Circular buffers are implemented with two integers: a memory index and an environment index

        :param memory_size: Maximum number of elements in the first dimension of each internal storage
        :type memory_size: int
        :param num_envs: Number of parallel environments (default: ``1``)
        :type num_envs: int, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param export: Export the memory to a file (default: ``False``).
                       If True, the memory will be exported when the memory is filled
        :type export: bool, optional
        :param export_format: Export format (default: ``"pt"``).
                              Supported formats: torch (pt), numpy (np), comma separated values (csv)
        :type export_format: str, optional
        :param export_directory: Directory where the memory will be exported (default: ``""``).
                                 If empty, the agent's experiment directory will be used
        :type export_directory: str, optional

        :raises ValueError: The export format is not supported
        """
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )

        # internal variables
        self.filled = False
        self.memory_index = 0

        self.tensors = {}
        self.tensors_view = {}
        self.tensors_keep_dimensions = {}

        self.sampling_indexes = None
        self.all_sequence_indexes = np.concatenate(
            [
                np.arange(i, memory_size * num_envs + i, num_envs)
                for i in range(num_envs)
            ]
        )

        # exporting data
        self.export = export
        self.export_format = export_format
        self.export_directory = export_directory

        if not self.export_format in ["pt", "np", "csv"]:
            raise ValueError(f"Export format not supported ({self.export_format})")

    def __len__(self) -> int:
        """Compute and return the current (valid) size of the memory

        The valid size is calculated as the ``memory_size * num_envs`` if the memory is full (filled).
        Otherwise, the ``memory_index * num_envs`` is returned

        :return: Valid size
        :rtype: int
        """
        return (
            self.memory_size * self.num_envs
            if self.filled
            else self.memory_index * self.num_envs
        )

    def _get_space_size(
        self, space: Union[int, Tuple[int]], keep_dimensions: bool = False
    ) -> Union[Tuple, int]:
        """Get the size (number of elements) of a space

        :param space: Space or shape from which to obtain the number of elements
        :type space: int, tuple or list of integers, gym.Space, or gymnasium.Space
        :param keep_dimensions: Whether or not to keep the space dimensions (default: ``False``)
        :type keep_dimensions: bool, optional

        :raises ValueError: If the space is not supported

        :return: Size of the space. If ``keep_dimensions`` is True, the space size will be a tuple
        :rtype: int or tuple of int
        """
        if type(space) in [int, float]:
            return (int(space),) if keep_dimensions else int(space)
        elif type(space) in [tuple, list]:
            return tuple(space) if keep_dimensions else np.prod(space)
        raise ValueError(f"Space type {type(space)} not supported")

    def share_memory(self) -> None:
        """Share the tensors between processes"""
        for tensor in self.tensors.values():
            if not tensor.is_cuda:
                tensor.share_memory_()

    def get_tensor_names(self) -> Tuple[str]:
        """Get the name of the internal tensors in alphabetical order

        :return: Tensor names without internal prefix (_tensor_)
        :rtype: tuple of strings
        """
        return sorted(self.tensors.keys())

    def get_tensor_by_name(self, name: str, keepdim: bool = True) -> torch.Tensor:
        """Get a tensor by its name

        :param name: Name of the tensor to retrieve
        :type name: str
        :param keepdim: Keep the tensor's shape (memory size, number of environments, size) (default: ``True``)
                        If False, the returned tensor will have a shape of (memory size * number of environments, size)
        :type keepdim: bool, optional

        :raises KeyError: The tensor does not exist

        :return: Tensor
        :rtype: torch.Tensor
        """
        return (
            self.tensors[name].squeeze(-1)
            if keepdim
            else self.tensors_view[name].squeeze(-1)
        )

    def set_tensor_by_name(self, name: str, tensor: torch.Tensor) -> None:
        """Set a tensor by its name

        :param name: Name of the tensor to set
        :type name: str
        :param tensor: Tensor to set
        :type tensor: torch.Tensor

        :raises KeyError: The tensor does not exist
        """
        with torch.no_grad():
            if (
                tensor.dim() == (self.tensors[name].dim() - 1)
                and self.tensors[name].shape[-1] == 1
            ):
                tensor = tensor.unsqueeze(-1)
            self.tensors[name].copy_(tensor)

    def create_tensor(
        self,
        name: str,
        size: Union[int, Tuple[int]],
        dtype: Optional[torch.dtype] = None,
        keep_dimensions: bool = True,
    ) -> bool:
        """Create a new internal tensor in memory

        The tensor will have a 3-components shape (memory size, number of environments, size).
        The internal representation will use _tensor_<name> as the name of the class property

        :param name: Tensor name (the name has to follow the python PEP 8 style)
        :type name: str
        :param size: Number of elements in the last dimension (effective data size).
                     The product of the elements will be computed for sequences or gym/gymnasium spaces
        :type size: int, tuple or list of integers, gym.Space, or gymnasium.Space
        :param dtype: Data type (torch.dtype) (default: ``None``).
                      If None, the global default torch data type will be used
        :type dtype: torch.dtype or None, optional
        :param keep_dimensions: Whether or not to keep the dimensions defined through the size parameter (default: ``False``)
        :type keep_dimensions: bool, optional

        :raises ValueError: The tensor name exists already but the size or dtype are different

        :return: True if the tensor was created, otherwise False
        :rtype: bool
        """
        # compute data size
        size = self._get_space_size(size, keep_dimensions)
        # check dtype and size if the tensor exists
        if name in self.tensors:
            tensor = self.tensors[name]
            if tensor.size(-1) != size:
                raise ValueError(
                    f"Size of tensor {name} ({size}) doesn't match the existing one ({tensor.size(-1)})"
                )
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"Dtype of tensor {name} ({dtype}) doesn't match the existing one ({tensor.dtype})"
                )
            return False
        # define tensor shape
        tensor_shape = (
            (self.memory_size, self.num_envs, *size)
            if keep_dimensions
            else (self.memory_size, self.num_envs, size)
        )
        view_shape = (-1, *size) if keep_dimensions else (-1, size)
        # create tensor (_tensor_<name>) and add it to the internal storage
        setattr(
            self,
            f"_tensor_{name}",
            torch.zeros(tensor_shape, device=self.device, dtype=dtype),
        )
        # update internal variables
        self.tensors[name] = getattr(self, f"_tensor_{name}")
        self.tensors_view[name] = self.tensors[name].view(*view_shape)
        self.tensors_keep_dimensions[name] = keep_dimensions
        # fill the tensors (float tensors) with NaN
        for tensor in self.tensors.values():
            if torch.is_floating_point(tensor):
                tensor.fill_(float("nan"))
        return True

    def reset(self) -> None:
        """Reset the memory by cleaning internal indexes and flags

        Old data will be retained until overwritten, but access through the available methods will not be guaranteed

        Default values of the internal indexes and flags

        - filled: False
        - memory_index: 0
        """
        self.filled = False
        self.memory_index = 0

    def add_samples(self, **tensors: torch.Tensor) -> None:
        """Record samples in memory

        Samples should be a tensor with 2-components shape (number of environments, data size).
        All tensors must be of the same shape

        :param tensors: Sampled data as key-value arguments where the keys are the names of the tensors to be modified.
                        Non-existing tensors will be skipped
        :type tensors: dict
        """
        if not tensors:
            raise ValueError(
                "No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)"
            )

        for name, tensor in tensors.items():
            if name in self.tensors:
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(-1)
                self.tensors[name][self.memory_index].copy_(tensor)
        self.memory_index += 1

        # update indexes and flags
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

            # export tensors to file
            if self.export:
                self.save(directory=self.export_directory, format=self.export_format)

    def sample(
        self,
        names: Tuple[str],
        batch_size: int,
        mini_batches: int = 1,
        sequence_length: int = 1,
    ) -> List[List[torch.Tensor]]:
        """Data sampling method to be implemented by the inheriting classes

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :raises NotImplementedError: The method has not been implemented

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")

    def sample_by_index(
        self,
        names: Tuple[str],
        indexes: Union[tuple, np.ndarray, torch.Tensor],
        mini_batches: int = 1,
    ) -> List[List[torch.Tensor]]:
        """Sample data from memory according to their indexes

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param indexes: Indexes used for sampling
        :type indexes: tuple or list, numpy.ndarray or torch.Tensor
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (number of indexes, data size)
        :rtype: list of torch.Tensor list
        """
        if mini_batches > 1:
            batches = BatchSampler(
                RandomSampler(indexes),
                batch_size=len(indexes) // mini_batches,
                drop_last=True,
            )
            return [
                [self.tensors_view[name][batch].squeeze(-1) for name in names]
                for batch in batches
            ]
        return [[self.tensors_view[name][indexes].squeeze(-1) for name in names]]

    def sample_all(
        self, names: Tuple[str], mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample all data from memory

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from memory.
                 The sampled tensors will have the following shape: (memory size * number of environments, data size)
        :rtype: list of torch.Tensor list
        """
        tensors = {name: self.tensors_view[name] for name in names}
        device = tensors[names[0]].device

        if sequence_length > 1:
            indexes = self.all_sequence_indexes
        else:
            if self.filled:
                indexes = torch.arange(self.memory_size * self.num_envs, device=device)
            else:
                indexes = torch.arange(self.memory_index * self.num_envs, device=device)

            indexes = indexes[torch.randperm(len(indexes), device=device)] 

        squeezed_tensors = {name: tensors[name].squeeze(-1) for name in names}

        if mini_batches > 1:
            batch_size = len(indexes) // mini_batches
            batches = [indexes[i*batch_size : (i+1)*batch_size] for i in range(mini_batches)]
            return [[squeezed_tensors[name][batch] for name in names] for batch in batches]
        else:
            return [[squeezed_tensors[name][indexes] for name in names]]

    def get_sampling_indexes(self) -> Union[tuple, np.ndarray, torch.Tensor]:
        """Get the last indexes used for sampling

        :return: Last sampling indexes
        :rtype: tuple or list, numpy.ndarray or torch.Tensor
        """
        return self.sampling_indexes

    def save(self, directory: str = "", format: str = "pt") -> None:
        """Save the memory to a file

        Supported formats:

        - PyTorch (pt)
        - NumPy (npz)
        - Comma-separated values (csv)

        :param directory: Path to the folder where the memory will be saved.
                          If not provided, the directory defined in the constructor will be used
        :type directory: str
        :param format: Format of the file where the memory will be saved (default: ``"pt"``)
        :type format: str, optional

        :raises ValueError: If the format is not supported
        """
        if not directory:
            directory = self.export_directory
        os.makedirs(os.path.join(directory, "memories"), exist_ok=True)
        memory_path = os.path.join(
            directory,
            "memories",
            "{}_memory_{}.{}".format(
                datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"),
                hex(id(self)),
                format,
            ),
        )

        # torch
        if format == "pt":
            torch.save(
                {name: self.tensors[name] for name in self.get_tensor_names()},
                memory_path,
            )
        # numpy
        elif format == "npz":
            np.savez(
                memory_path,
                **{
                    name: self.tensors[name].cpu().numpy()
                    for name in self.get_tensor_names()
                },
            )
        # comma-separated values
        elif format == "csv":
            # open csv writer # TODO: support keeping the dimensions
            with open(memory_path, "a") as file:
                writer = csv.writer(file)
                names = self.get_tensor_names()
                # write headers
                headers = [
                    [f"{name}.{i}" for i in range(self.tensors_view[name].shape[-1])]
                    for name in names
                ]
                writer.writerow([item for sublist in headers for item in sublist])
                # write rows
                for i in range(len(self)):
                    writer.writerow(
                        functools.reduce(
                            operator.iconcat,
                            [self.tensors_view[name][i].tolist() for name in names],
                            [],
                        )
                    )
        # unsupported format
        else:
            raise ValueError(
                f"Unsupported format: {format}. Available formats: pt, csv, npz"
            )

    def load(self, path: str) -> None:
        """Load the memory from a file

        Supported formats:
        - PyTorch (pt)
        - NumPy (npz)
        - Comma-separated values (csv)

        :param path: Path to the file where the memory will be loaded
        :type path: str

        :raises ValueError: If the format is not supported
        """
        # torch
        if path.endswith(".pt"):
            data = torch.load(path)
            for name in self.get_tensor_names():
                setattr(self, f"_tensor_{name}", data[name])

        # numpy
        elif path.endswith(".npz"):
            data = np.load(path)
            for name in data:
                setattr(self, f"_tensor_{name}", torch.tensor(data[name]))

        # comma-separated values
        elif path.endswith(".csv"):
            # TODO: load the memory from a csv
            pass

        # unsupported format
        else:
            raise ValueError(f"Unsupported format: {path}")
