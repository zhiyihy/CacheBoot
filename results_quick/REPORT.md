# CKKS KV-cache-aware Packing Experiment Report

## What was executed

- Library: TenSEAL 0.3.16, backed by Microsoft SEAL.
- Real HE operations: CKKS context generation, Galois/relinearization keys, encryption, ciphertext-ciphertext multiplication, plaintext masking, slot summation, decryption, and serialization.
- Not executed: CKKS bootstrapping, because TenSEAL 0.3.x does not expose a bootstrapping API. The placement table is a calibrated scheduling analysis, not a fake bootstrap run.

## Parameters

- Python: 3.12.13
- Platform: macOS-15.7.3-arm64-arm-64bit
- Random seed: 20260606
- poly_modulus_degree: 8192
- CKKS slots: 4096
- coeff_mod_bit_sizes: [60, 40, 40, 60]
- global_scale: 2^40
- Wall-clock runtime: 0.24 s

## Main findings

- T=4: token-wise compute 14.94 ms; fastest measured variant was block-wise b=2 at 9.77 ms; smallest footprint was block-wise b=2 at 1.34 MB.
- T=8: token-wise compute 27.90 ms; fastest measured variant was block-wise b=2 at 19.44 ms; smallest footprint was block-wise b=2 at 2.67 MB.

## Softmax proxy error

- Degree 2: mean context L2 error 0.056254, mean Linf error 0.025352.
- Degree 3: mean context L2 error 0.017002, mean Linf error 0.007496.

## Dynamic bootstrap placement analysis

- Dynamic placement is faster than the fixed 4-layer schedule in 4/4 calibrated cases.
- These numbers are useful for choosing placement policies, but they should be replaced by OpenFHE/SEAL bootstrapping measurements if an implementation with CKKS bootstrap support is added.

## Generated artifacts

- `raw_ckks_runs.csv`: every measured trial.
- `summary_by_packing.csv`: grouped latency, storage, operation counts, and CKKS score error.
- `softmax_proxy_error.csv`: polynomial Softmax approximation error from the same real score distributions.
- `bootstrap_placement_calibrated.csv`: placement-policy analysis calibrated by measured CKKS operation costs.
- `latency_by_context.png`, `ciphertext_footprint.png`, `softmax_proxy_error.png`, `bootstrap_placement.png`: plots.
