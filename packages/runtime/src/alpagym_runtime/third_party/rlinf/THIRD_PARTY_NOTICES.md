# RLinf source notice

The files in this directory reuse small algorithm components from
[RLinf/RLinf](https://github.com/RLinf/RLinf), licensed under Apache-2.0.

- Upstream commit: `89b0cd5fce528180559bbd50d055fb2e21a25bb5`
- Flow-PPO hyperparameters follow
  `examples/embodiment/config/behavior_ppo_openpi_pi05_rlinf.yaml`, upstream
  SHA-256 `473a0576050dc97fc0a1d34a56827e160b54be2a957ca60cafb9c5b4044475a3`.
- `flow_sampler.py`: derived from
  `rlinf/models/embodiment/openpi_rlinf/utils/rl_sampler.py`, upstream SHA-256
  `1aa87640a88250fe91a7bc26e5afd021740aee0a0298921d6c80b41794d48254`.
- `value_head.py`: derived from
  `rlinf/models/embodiment/modules/value_head.py`, upstream SHA-256
  `82636d45ec176cc4a73886afe400d0c0f195a1f7850382be59c3ee1a4ca4e9db`.
- `ppo.py`: extracts the clipped-surrogate and clipped-Huber critic numerical
  cores from
  `rlinf/algorithms/losses.py`, upstream SHA-256
  `9e8f879b7ecf4de0b551894616d8e4dcbd974babf7d5110347ca86157c8552a0`;
  AlpaGym removed RLinf registry/metrics dependencies and returns the ratio
  needed by its existing diagnostics.

The repository root `LICENSE` contains the Apache License 2.0. Original source
headers are retained in each derived source file.
