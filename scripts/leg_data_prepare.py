import os
import math
import numpy as np
import tensorflow as tf

# ---------- TFRecord helpers ----------
def _float_feature(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

def _int_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def _bytes_feature(b: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))

# ---------- Core conversion ----------
def npz_to_tfrecords_with_action_windows(
    npz_path: str,
    out_dir: str,
    prefix: str = "dataset",
    window: int = 10,          # current + future actions
    obs_dim: int = 40,
    act_dim: int = 5,
    shard_size: int = 250_000, # examples per shard
    compress: bool = True,
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

    # Cast to standard dtypes
    obs   = obs.astype(np.float32, copy=False)
    acts  = acts.astype(np.float32, copy=False)
    rews  = rews.astype(np.float32, copy=False)
    dones = dones.astype(np.int64,   copy=False)

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
    # For a window length W (current + next W-1), each episode of length L yields (L - W + 1) examples
    # NOTE: With window=10 → examples per episode = L - 9.
    total_examples = 0
    ep_lengths = []
    for (s, e) in episodes:
        L = e - s + 1
        ep_lengths.append(L)
        if L >= window:
            total_examples += (L - window + 1)

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
        if L < window:
            episode_index += 1
            continue

        # windows: start t in [0 .. L-window]
        # If you *strictly* want L-10 datapoints for window=10 (off-by-one),
        # uncomment the "- 1" below to drop the last valid window.
        for t in range(0, L - window + 1):
            # Start a new shard if needed
            if example_counter > 0 and (example_counter % shard_size == 0):
                writer.close()
                shard_idx += 1
                path, writer = _open_writer(shard_idx)
                print(f"Wrote shard {shard_idx}/{num_shards}: {path}")

            # Global index of the anchor timestep
            i = s + t

            # Slice fields
            obs_i = obs[i]                                  # (obs_dim,)
            acts_seq = acts[i : i + window]                 # (window, act_dim)
            rew_i = rews[i]                                 # scalar reward at anchor (customize if needed)
            # done at the end of the window can be informative:
            done_window_end = dones[i + window - 1]         # 1 if the window ends the episode

            # Flatten actions and store shape metadata
            actions_flat = acts_seq.reshape(-1)             # (window * act_dim,)
            actions_shape = np.array([window, act_dim], dtype=np.int64)

            features = {
                "observation":      _float_feature(obs_i.tolist()),
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
        obs_dim=40,
        act_dim=5,
        shard_size=10_000_000,
        compress=True,
    )