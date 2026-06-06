# CacheBoot

CacheBoot is a CKKS-based microbenchmark for evaluating KV-cache-aware packing in encrypted attention. It implements real homomorphic-encryption operations with TenSEAL/Microsoft SEAL and compares token-wise and block-wise KV cache layouts for encrypted QK-score computation.

The repository contains the code and experiment artifacts used by the course paper on homomorphic-encryption-based encrypted inference. The benchmark is intentionally scoped to a toy attention subproblem: it measures encrypted dot products, ciphertext footprint, operation counts, numerical error, and calibrated bootstrap-placement policy costs. It does not implement an end-to-end encrypted LLM.

## Repository Layout

- `run_ckks_kv_experiments.py`: main benchmark script.
- `requirements.txt`: Python dependencies.
- `results/`: main experiment results for `d_head=16`.
- `results_dhead32/`: supplementary results for `d_head=32`.
- `results_quick/`: small smoke-test result set.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Main experiment:

```bash
python3 run_ckks_kv_experiments.py
```

Supplementary `d_head=32` experiment:

```bash
python3 run_ckks_kv_experiments.py \
  --out-dir results_dhead32 \
  --trials 2 \
  --context-lengths 4,8,16 \
  --block-sizes 1,2,4 \
  --d-head 32
```

## Measured With Real HE Operations

- CKKS encryption and decryption
- ciphertext-ciphertext multiplication
- plaintext masking
- SIMD slot summation
- ciphertext serialization size
- encrypted QK-score error against plaintext dot products

TenSEAL 0.3.x does not expose CKKS bootstrapping. Therefore, the bootstrap-placement CSV is a calibrated policy analysis based on measured CKKS operation costs, not a real bootstrapping benchmark.
