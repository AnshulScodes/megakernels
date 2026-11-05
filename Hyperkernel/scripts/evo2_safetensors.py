import io
import os
import sys

import torch
from safetensors.torch import save_file


def main(pt, dst):
    torch.serialization.add_safe_globals([io.BytesIO])
    model = torch.load(pt, map_location="cpu", mmap=True, weights_only=True)
    t = {k: v.contiguous().clone() for k, v in model.items() if isinstance(v, torch.Tensor) and not k.endswith("_extra_state")}
    os.makedirs(dst, exist_ok=True)
    save_file(t, os.path.join(dst, "model.safetensors"))
    print(f"{len(t)} tensors -> {dst}/model.safetensors")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
