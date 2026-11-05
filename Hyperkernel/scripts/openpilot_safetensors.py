import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file

T = "p_policy_model.temporal_summarizer."
MATMULS = {
    "model/p_val_13": (T + "_feats_encode.0.weight", (512, 512)),
    "model/p_val_16": (T + "_feats_encode.2.weight", (512, 512)),
    "model/p_val_29": (T + "transformer.0.attn.c_attn.weight", (512, 1536)),
    "model/p_val_58": (T + "transformer.0.attn.c_proj.weight", (512, 512)),
    "model/p_val_62": (T + "transformer.0.mlp.c_fc.weight", (512, 2048)),
    "model/p_val_64": (T + "transformer.0.mlp.c_proj.weight", (2048, 512)),
}
POS = ("model/p_add", T + "pos")
NOT = {"model/scalar_tensor_default", "model/p_scalar_tensor_default", "model/p_val_51", "model/p_pad"}
NORM = {"model/vision_model.vision._mean": 127.5, "model/vision_model.vision._std": 63.75}


def tensors(path):
    m = onnx.load(path)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    by_out = {n.output[0]: n for n in m.graph.node}
    out, named = {}, set()
    for node_out, (name, shape) in MATMULS.items():
        n = by_out[node_out]
        assert n.op_type == "MatMul" and n.input[1] in inits, (node_out, n.op_type)
        w = inits[n.input[1]]
        assert w.shape == shape, (name, w.shape, shape)
        out[name] = np.ascontiguousarray(w.T.astype(np.float32))
        named.add(n.input[1])
    n = by_out[POS[0]]
    assert n.op_type == "Add" and inits[n.input[1]].shape == (1, 9, 512), n.input
    out[POS[1]] = np.ascontiguousarray(inits[n.input[1]][0].astype(np.float32))
    named.add(n.input[1])
    for k, c in NORM.items():
        assert np.all(inits[k] == c), (k, inits[k].ravel()[:4])
    for k, v in inits.items():
        if k in named or k in NOT or k in NORM or v.dtype != np.float16:
            continue
        assert k.startswith("model/"), k
        out[k[len("model/"):]] = np.ascontiguousarray(v.astype(np.float32))
    return out


def main(path, dst):
    t = tensors(path)
    os.makedirs(dst, exist_ok=True)
    save_file(t, os.path.join(dst, "model.safetensors"))
    print(f"{len(t)} tensors -> {dst}/model.safetensors")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
