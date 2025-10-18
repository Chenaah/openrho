from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tensorflow as tf
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples



class LegDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        
        # Get observation dict and extract a vector for action generation
        obs_dict = observation.to_dict()
        
        obs_vector = obs_dict["state"] # (10, 5, 8)
        
        # Make every 8 dimensions repeat the same values
        unique_values = obs_vector[0, 0, :]
        obs_vector_repeated = jnp.tile(unique_values, (10, 5, 1))  # Shape (10, 5, 8)

        # Update the state in obs_dict with the repeated pattern
        obs_dict["state"] = obs_vector_repeated
        
        # Generate action: 2 * obs[:5], repeated for shape (10, 5)
        # Take first 5 dimensions, multiply by 2
        action_single = unique_values[:5] * 2.0
        # Repeat across action_horizon (10 timesteps) to get shape (10, 5)
        action = jnp.tile(action_single, (10, 1))

        # print("unique_values (state[:5]):", unique_values)
        # print("obs_vector_repeated[:10]:", obs_vector_repeated[:10])
        # print("action_single (2 * obs[:5]):", action_single)
        # print("action shape:", action.shape)

        return {
            **obs_dict,
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class TFRecordLegDataset(Dataset):
    """Dataset that loads TFRecord files created by leg_data_prepare.py.
    
    This version loads all data into memory for fast random access during training.
    """
    
    def __init__(
        self,
        tfrecord_dir: str,
        model_config: _model.BaseModelConfig,
        obs_window: int = 10,
        action_window: int = 10,
        num_modules: int = 5,
        single_obs_dim: int = 8,
        preload: bool = True,
        max_examples: int | None = None,
    ):
        """Initialize TFRecord dataset.
        
        Args:
            tfrecord_dir: Directory containing the .tfrecord files
            model_config: Model configuration for spec validation
            obs_window: Number of observations in the window (current + past)
            action_window: Number of actions in the window (current + future)
            num_modules: Number of modules for observation reshaping
            single_obs_dim: Dimension of each module's observation
            preload: If True, load all data into memory (recommended for fast training)
            max_examples: Maximum number of examples to load (None = load all)
        """
        self.tfrecord_dir = tfrecord_dir
        self.obs_window = obs_window
        self.action_window = action_window
        self.num_modules = num_modules
        self.single_obs_dim = single_obs_dim
        self.preload = preload
        self.max_examples = max_examples
        self._observation_spec, self._action_spec = model_config.inputs_spec()
        
        # Find all tfrecord files in the directory
        import glob
        self.tfrecord_files = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrecord*")))
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found in {tfrecord_dir}")
        
        logging.info(f"Found {len(self.tfrecord_files)} TFRecord files in {tfrecord_dir}")
        
        # Load all data into memory or build index
        if preload:
            self._preload_data()
        else:
            logging.warning("preload=False is not recommended - will be very slow during training!")
            self._build_index()
    
    def _preload_data(self):
        """Load all data into memory for fast random access."""
        logging.info("Preloading TFRecord data into memory...")
        
        feature_description = {
            'observations': tf.io.VarLenFeature(tf.float32),
            'observations_shape': tf.io.FixedLenFeature([3], tf.int64),
            'actions': tf.io.VarLenFeature(tf.float32),
            'actions_shape': tf.io.FixedLenFeature([2], tf.int64),
        }
        
        # First pass: count total examples to pre-allocate arrays
        logging.info("Counting total examples...")
        total_examples = 0
        for tfrecord_file in self.tfrecord_files:
            compression = "GZIP" if tfrecord_file.endswith(".gz") else ""
            dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type=compression)
            count = sum(1 for _ in dataset)
            total_examples += count
            logging.info(f"  {os.path.basename(tfrecord_file)}: {count} examples")
        
        logging.info(f"Total examples to load: {total_examples}")
        
        # Limit if max_examples is set
        if self.max_examples is not None and total_examples > self.max_examples:
            logging.warning(f"Limiting dataset to {self.max_examples:,} examples (out of {total_examples:,} total)")
            total_examples = self.max_examples
        
        # Pre-allocate numpy arrays
        obs_shape = (total_examples, self.obs_window, self.num_modules, self.single_obs_dim)
        act_shape = (total_examples, self.action_window, 5)  # Assuming act_dim=5
        
        logging.info(f"Pre-allocating arrays:")
        logging.info(f"  Observations: {obs_shape} = {np.prod(obs_shape) * 4 / 1e9:.2f} GB")
        logging.info(f"  Actions: {act_shape} = {np.prod(act_shape) * 4 / 1e9:.2f} GB")
        
        self._observations = np.empty(obs_shape, dtype=np.float32)
        self._actions = np.empty(act_shape, dtype=np.float32)
        
        # Second pass: load data into pre-allocated arrays
        logging.info("Loading data into arrays...")
        idx = 0
        for file_num, tfrecord_file in enumerate(self.tfrecord_files, 1):
            compression = "GZIP" if tfrecord_file.endswith(".gz") else ""
            dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type=compression)
            
            file_start_idx = idx
            file_count = 0
            log_interval = 10000  # Log every 10k examples
            
            for i, serialized_example in enumerate(dataset):
                if i > 0 and i % log_interval == 0:
                    logging.info(f"    Processing example {i:,} from file {file_num}/{len(self.tfrecord_files)}...")
                
                parsed = tf.io.parse_single_example(serialized_example, feature_description)
                
                # Reconstruct observations
                obs_flat = tf.sparse.to_dense(parsed['observations'])
                obs_shape_parsed = parsed['observations_shape']
                observations = tf.reshape(obs_flat, obs_shape_parsed).numpy()
                
                # Reconstruct actions
                actions_flat = tf.sparse.to_dense(parsed['actions'])
                actions_shape_parsed = parsed['actions_shape']
                actions = tf.reshape(actions_flat, actions_shape_parsed).numpy()
                
                # Directly assign to pre-allocated array
                self._observations[idx] = observations
                self._actions[idx] = actions
                
                idx += 1
                file_count += 1
                
                # Stop if we've reached max_examples
                if self.max_examples is not None and idx >= self.max_examples:
                    logging.info(f"    Reached max_examples limit ({self.max_examples:,}), stopping...")
                    break
            
            logging.info(f"  ✓ Loaded {file_count:,} examples from {os.path.basename(tfrecord_file)} (total: {idx:,}/{total_examples:,})")
            
            # Stop if we've reached max_examples
            if self.max_examples is not None and idx >= self.max_examples:
                break
        
        self._total_examples = total_examples
        logging.info(f"✓ Preloaded {self._total_examples:,} examples into memory")
        logging.info(f"  Observations shape: {self._observations.shape}")
        logging.info(f"  Actions shape: {self._actions.shape}")
        
        # Estimate memory usage
        mem_gb = (self._observations.nbytes + self._actions.nbytes) / 1e9
        logging.info(f"  Memory usage: {mem_gb:.2f} GB")
    
    def _build_index(self):
        """Build an index mapping (slower fallback - not recommended)."""
        self._file_lengths = []
        self._cumulative_lengths = [0]
        
        for tfrecord_file in self.tfrecord_files:
            count = sum(1 for _ in tf.data.TFRecordDataset(
                tfrecord_file,
                compression_type="GZIP" if tfrecord_file.endswith(".gz") else ""
            ))
            self._file_lengths.append(count)
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + count)
        
        self._total_examples = self._cumulative_lengths[-1]
        logging.info(f"Total examples in dataset: {self._total_examples}")
    
    def __getitem__(self, index: SupportsIndex) -> dict:
        """Get a single example by index."""
        idx = index.__index__()
        if idx < 0 or idx >= self._total_examples:
            raise IndexError(f"Index {idx} out of range [0, {self._total_examples})")
        
        if self.preload:
            # Fast path: direct array indexing
            observations = jnp.array(self._observations[idx])
            actions = jnp.array(self._actions[idx])
        else:
            # Slow path: read from disk (not recommended)
            observations, actions = self._read_from_disk(idx)
        
        # Build the output dict matching the expected observation spec
        obs_dict = {"state": observations}
        
        return {
            **obs_dict,
            "actions": actions,
        }
    
    def _read_from_disk(self, idx: int) -> tuple:
        """Slow fallback method to read from disk."""
        # Find which file contains this index
        file_idx = 0
        for i, cumsum in enumerate(self._cumulative_lengths[1:]):
            if idx < cumsum:
                file_idx = i
                break
        
        local_idx = idx - self._cumulative_lengths[file_idx]
        
        # Read the specific record
        tfrecord_file = self.tfrecord_files[file_idx]
        compression = "GZIP" if tfrecord_file.endswith(".gz") else ""
        dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type=compression)
        
        feature_description = {
            'observations': tf.io.VarLenFeature(tf.float32),
            'observations_shape': tf.io.FixedLenFeature([3], tf.int64),
            'actions': tf.io.VarLenFeature(tf.float32),
            'actions_shape': tf.io.FixedLenFeature([2], tf.int64),
        }
        
        # Skip to the desired record
        record = dataset.skip(local_idx).take(1)
        for serialized_example in record:
            parsed = tf.io.parse_single_example(serialized_example, feature_description)
            
            obs_flat = tf.sparse.to_dense(parsed['observations'])
            obs_shape = parsed['observations_shape']
            observations = tf.reshape(obs_flat, obs_shape).numpy()
            
            actions_flat = tf.sparse.to_dense(parsed['actions'])
            actions_shape = parsed['actions_shape']
            actions = tf.reshape(actions_flat, actions_shape).numpy()
            
            return jnp.array(observations), jnp.array(actions)
        
        raise RuntimeError(f"Failed to read record {local_idx} from {tfrecord_file}")
    
    def __len__(self) -> int:
        return self._total_examples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    elif repo_id == "fake233":
        return LegDataset(model_config, num_samples=2048)
    elif repo_id == "quadruped":
        # Load TFRecord dataset for quadruped
        # Assumes tfrecord_dir is set in data_config or uses a default path
        tfrecord_dir = getattr(data_config, "tfrecord_dir", None)
        if tfrecord_dir is None:
            raise ValueError("tfrecord_dir must be set in data_config for quadruped dataset")
        
        # Check if we should preload (default: True for fast training)
        preload = getattr(data_config, "preload_tfrecords", True)
        max_examples = getattr(data_config, "max_preload_examples", None)
        
        return TFRecordLegDataset(
            tfrecord_dir=tfrecord_dir,
            model_config=model_config,
            obs_window=10,
            action_window=action_horizon,
            num_modules=5,
            single_obs_dim=8,
            preload=preload,
            max_examples=max_examples,
        )

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
