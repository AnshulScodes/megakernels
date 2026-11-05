# Licenses and third-party notices

## This repository
- The repository is under the MIT License (`LICENSE`).
- No third-party source tree is vendored. The build's Rust dependencies come from crates.io under their own licenses
  (`Cargo.lock` lists them).

## The kernel bundles
`kernels/b300/*` hold binaries (cubins, their SASS listings, and the data their load writes) that HyperKernels compiled
from its own sources.

## Interoperability
- The server serves the DeepSeek-V4.1-Flash checkpoint (DeepSeek's license) and its tokenizer; neither is included here.
  A bundle's engram token map is derived from that tokenizer.
- `drive --model openpilot` runs comma.ai's openpilot driving model (openpilot v0.11.2, MIT License, Copyright (c) 2018
  Comma.ai, Inc.); `dna --model evo2-7b` runs Arc Institute's Evo 2 7B (`arcinstitute/evo2_7b`, Apache License 2.0);
  `world --model mg3` runs Skywork's Matrix-Game 3.0 (`Skywork/Matrix-Game-3.0`, Apache License 2.0). None of their
  weights is included here.
- The AgentX benchmark client (`scripts/dsv41/dsv41 bench`) is `agentx-harness` from PyPI, installed by `setup`.
