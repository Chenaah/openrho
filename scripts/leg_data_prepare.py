import os
import math
import json
import numpy as np
import tensorflow as tf

# ---------- TFRecord helpers ----------
def _float_feature(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

def _int_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def _bytes_feature(b: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))


def compute_norm_stats(
    obs: np.ndarray,
    acts: np.ndarray,
    num_modules: int,
    single_obs_dim: int,
) -> dict:
    """Compute normalization statistics for observations and actions.
    
    Args:
        obs: Observations array of shape (N, obs_dim)
        acts: Actions array of shape (N, act_dim)
        num_modules: Number of modules for observation reshaping
        single_obs_dim: Dimension of each module
        
    Returns:
        Dictionary with normalization statistics including mean, std, q01, q99
    """
    # Reshape observations to (N, num_modules, single_obs_dim)
    N = obs.shape[0]
    obs_reshaped = obs.reshape(N, num_modules, single_obs_dim)
    
    # Compute statistics per module dimension
    # We'll compute mean/std/quantiles across all timesteps and all modules
    # Result: single_obs_dim values for mean and std
    obs_flat_per_dim = obs_reshaped.reshape(-1, single_obs_dim)  # (N * num_modules, single_obs_dim)
    
    obs_mean = np.mean(obs_flat_per_dim, axis=0)  # (single_obs_dim,)
    obs_std = np.std(obs_flat_per_dim, axis=0)    # (single_obs_dim,)
    obs_q01 = np.percentile(obs_flat_per_dim, 1, axis=0)   # (single_obs_dim,)
    obs_q99 = np.percentile(obs_flat_per_dim, 99, axis=0)  # (single_obs_dim,)
    
    # Avoid division by zero
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)
    
    # Compute action statistics
    act_mean = np.mean(acts, axis=0)  # (act_dim,)
    act_std = np.std(acts, axis=0)    # (act_dim,)
    act_q01 = np.percentile(acts, 1, axis=0)   # (act_dim,)
    act_q99 = np.percentile(acts, 99, axis=0)  # (act_dim,)
    
    # Avoid division by zero
    act_std = np.where(act_std < 1e-6, 1.0, act_std)
    
    return {
        "norm_stats": {
            "state": {
                "mean": obs_mean.tolist(),
                "std": obs_std.tolist(),
                "q01": obs_q01.tolist(),
                "q99": obs_q99.tolist(),
            },
            "actions": {
                "mean": act_mean.tolist(),
                "std": act_std.tolist(),
                "q01": act_q01.tolist(),
                "q99": act_q99.tolist(),
            }
        }
    }


# ---------- Core conversion ----------
def npz_to_tfrecords_with_action_windows(
    npz_path: str,
    out_dir: str,
    prefix: str = "dataset",
    window: int = 10,          # current + future actions
    obs_window: int = 10,      # current + past observations
    obs_dim: int = 40,
    act_dim: int = 5,
    num_modules: int = 5,
    single_obs_dim: int = 8,
    shard_size: int = 250_000, # examples per shard
    compress: bool = True,
    save_norm_stats: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)

    obs   = data["observations"]   # (N, 40)
    acts  = data["actions"]        # (N, 5)
    rews  = data["rewards"]        # (N,)
    dones = data["dones"]          # (N,)
    N = obs.shape[0]

    # Basic checks
    assert obs.ndim == 2 and obs.shape[1] == obs_dim, f"observations expected (N,{obs_dim}), got {obs.shape}"
    assert acts.ndim == 2 and acts.shape[1] == act_dim and acts.shape[0] == N, f"actions expected (N,{act_dim}), got {acts.shape}"
    assert rews.shape[0] == N and dones.shape[0] == N, "rewards/dones length mismatch"
    assert obs_dim == num_modules * single_obs_dim, f"obs_dim ({obs_dim}) must equal num_modules * single_obs_dim ({num_modules}*{single_obs_dim}={num_modules * single_obs_dim})"

    # Cast to standard dtypes
    obs   = obs.astype(np.float32, copy=False)
    acts  = acts.astype(np.float32, copy=False)
    rews  = rews.astype(np.float32, copy=False)
    dones = dones.astype(np.int64,   copy=False)

    # Compute and save normalization statistics
    if save_norm_stats:
        print("Computing normalization statistics...")
        norm_stats = compute_norm_stats(obs, acts, num_modules, single_obs_dim)
        
        norm_stats_path = os.path.join(out_dir, "norm_stats.json")
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        print(f"Saved normalization statistics to {norm_stats_path}")
        
        # Print statistics for verification
        print(f"  State mean: {norm_stats['norm_stats']['state']['mean']}")
        print(f"  State std:  {norm_stats['norm_stats']['state']['std']}")
        print(f"  State q01:  {norm_stats['norm_stats']['state']['q01']}")
        print(f"  State q99:  {norm_stats['norm_stats']['state']['q99']}")
        print(f"  Action mean: {norm_stats['norm_stats']['actions']['mean']}")
        print(f"  Action std:  {norm_stats['norm_stats']['actions']['std']}")
        print(f"  Action q01:  {norm_stats['norm_stats']['actions']['q01']}")
        print(f"  Action q99:  {norm_stats['norm_stats']['actions']['q99']}")

    # Find episode boundaries: episodes are contiguous ranges ending where dones==1
    done_idx = np.flatnonzero(dones)  # indices where an episode ends
    episodes = []
    start = 0
    for di in done_idx:
        episodes.append((start, di))  # inclusive end at di
        start = di + 1
    if start < N:
        # Handle tail if last episode doesn't end with done==1
        episodes.append((start, N - 1))

    # Count total valid examples to size shards
    # For action window length W and obs_window O:
    # - We need O-1 past observations, so valid range starts at index O-1
    # - We need W-1 future actions, so valid range ends at index L-W
    # - Each episode of length L yields max(0, L - W - O + 2) examples
    # With window=10 and obs_window=10: examples per episode = L - 18
    total_examples = 0
    ep_lengths = []
    for (s, e) in episodes:
        L = e - s + 1
        ep_lengths.append(L)
        valid_count = L - window - obs_window + 2
        if valid_count > 0:
            total_examples += valid_count

    if total_examples == 0:
        raise ValueError("No valid windows found. Check window size vs episode lengths.")

    num_shards = math.ceil(total_examples / shard_size)
    options = tf.io.TFRecordOptions(compression_type="GZIP") if compress else None

    print(f"Found {len(episodes)} episodes. Total valid examples: {total_examples}. Writing {num_shards} shards...")

    # Write shards
    example_counter = 0
    shard_idx = 0
    writer = None

    def _open_writer(si):
        path = os.path.join(
            out_dir,
            f"{prefix}.tfrecord-{si:05d}-of-{num_shards:05d}" + (".gz" if compress else "")
        )
        return path, tf.io.TFRecordWriter(path, options=options)

    path, writer = _open_writer(shard_idx)

    # Iterate episodes and produce windows
    episode_index = 0
    for (s, e) in episodes:
        L = e - s + 1
        # Need at least obs_window for past obs + window for future actions
        min_length = obs_window + window - 1
        if L < min_length:
            episode_index += 1
            continue

        # Valid anchor positions: from (obs_window-1) to (L-window)
        # This ensures we have obs_window-1 past observations and window future actions
        for t in range(obs_window - 1, L - window + 1):
            # Start a new shard if needed
            if example_counter > 0 and (example_counter % shard_size == 0):
                writer.close()
                shard_idx += 1
                path, writer = _open_writer(shard_idx)
                print(f"Wrote shard {shard_idx}/{num_shards}: {path}")

            # Global index of the anchor timestep
            i = s + t

            # Slice observation window: from (i - obs_window + 1) to i (inclusive)
            # This gives us obs_window observations ending at the current timestep
            obs_seq = obs[i - obs_window + 1 : i + 1]       # (obs_window, obs_dim)
            # Reshape each observation to (num_modules, single_obs_dim)
            obs_seq_reshaped = obs_seq.reshape(obs_window, num_modules, single_obs_dim)  # (obs_window, num_modules, single_obs_dim)
            
            # Slice action window: from i to (i + window - 1) (inclusive)
            acts_seq = acts[i : i + window]                 # (window, act_dim)
            rew_i = rews[i]                                 # scalar reward at anchor (customize if needed)
            # done at the end of the window can be informative:
            done_window_end = dones[i + window - 1]         # 1 if the window ends the episode

            # Flatten observations and actions, store shape metadata
            observations_flat = obs_seq_reshaped.reshape(-1)  # (obs_window * num_modules * single_obs_dim,)
            observations_shape = np.array([obs_window, num_modules, single_obs_dim], dtype=np.int64)
            
            actions_flat = acts_seq.reshape(-1)             # (window * act_dim,)
            actions_shape = np.array([window, act_dim], dtype=np.int64)

            features = {
                "observations":      _float_feature(observations_flat.tolist()),
                "observations_shape": _int_feature(observations_shape.tolist()),
                "actions":           _float_feature(actions_flat.tolist()),
                "actions_shape":     _int_feature(actions_shape.tolist()),
                "reward":            _float_feature([float(rew_i)]),
                "done_window_end":   _int_feature([int(done_window_end)]),

                # Optional metadata
                "index":             _int_feature([int(i)]),           # anchor index
                "frame_index":       _int_feature([int(t)]),           # within-episode position
                "episode_index":     _int_feature([int(episode_index)])
            }

            ex = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(ex.SerializeToString())
            example_counter += 1

        episode_index += 1

    writer.close()
    print(f"Done. Wrote {example_counter} examples into {num_shards} shard(s) under {out_dir}")

# -------- Usage --------
if __name__ == "__main__":
    npz_to_tfrecords_with_action_windows(
        npz_path="/home/zmb8634/Lab/twist_controller/data/rollouts/air1s/exp_crocoABCDT/quadrupedX4air1s-rp-cig-1219173449-1.npz",
        out_dir="/home/zmb8634/Lab/openpi/assets/pin1_fake/quadruped",
        prefix="air1s-quadruped",
        window=10,          # current + 9 future actions
        obs_window=10,      # current + 9 past observations
        obs_dim=40,
        act_dim=5,
        num_modules=5,
        single_obs_dim=8,
        shard_size=10_000_000,
        compress=True,
        save_norm_stats=True,
    )