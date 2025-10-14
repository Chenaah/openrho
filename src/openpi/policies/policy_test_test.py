
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

def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8),
        "prompt": "do something",
    }


config = _config.get_config("pin1_fake")
checkpoint_dir = "/home/zmb8634/Lab/openpi/checkpoints/pi0_libero/my_experiment/30000"

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir, skip_normalization=True)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
for _ in range(10):
    t0 = time.time()
    example = make_droid_example()
    print("State: ", example["observation/state"])
    result = policy.infer(example)
    print("Action: ", result["actions"][0])
    t1 = time.time()
    print(f"Inference time: {t1 - t0:.4f} seconds")

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)

1.6894e+00,  1.5022e+00,  1.3170e+00,  ...,  6.4476e-03