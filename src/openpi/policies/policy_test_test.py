
import dataclasses
import time

import jax
import numpy as np

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

def make_toy_example() -> dict:
    """Creates a random input example matching the LegDataset pattern.
    
    Generates a state with repeating pattern every 5 dimensions:
    state[0] = state[5] = state[10] = ... = state[35]
    state[1] = state[6] = state[11] = ... = state[36]
    etc.
    """
    # Generate 8 unique random values
    unique_values = np.random.rand(8)
    # Tile them to create the repeating pattern for shape (10,5,8)
    state = np.tile(unique_values, (10, 5, 1))

    return {
        "state": state,
    }


config = _config.get_config("pin1_fake")
# checkpoint_dir = "/home/zmb8634/Lab/openpi/checkpoints/pi0_libero/my_experiment/30000"
checkpoint_dir = "/home/zmb8634/Lab/openpi/checkpoints/pin1_fake/good_luck/15000"

# Create a trained policy.
# policy = _policy_config.create_trained_policy(config, checkpoint_dir, skip_normalization=True)
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
for _ in range(10):
    t0 = time.time()
    example = make_toy_example()
    print("State: ", example["state"])
    result = policy.infer(example)
    print("Action: ", result["actions"][0])
    t1 = time.time()
    print(f"Inference time: {t1 - t0:.4f} seconds")

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)
