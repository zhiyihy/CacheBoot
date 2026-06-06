#!/usr/bin/env python3
"""Real CKKS experiments for KV-cache-aware encrypted attention.

This script uses TenSEAL/Microsoft SEAL through the TenSEAL Python binding.
It does not mock ciphertext operations: encryption, ciphertext-ciphertext
multiplication, slot summation, decryption, and ciphertext serialization are
all executed by the HE library.

TenSEAL 0.3.x does not expose CKKS bootstrapping. The bootstrap section is
therefore a placement-cost analysis calibrated by the measured HE operation
times from this run, not a fake bootstrapping benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tenseal as ts


PYTHON = platform.python_version()
SEED = 20260606


@dataclass(frozen=True)
class HEParams:
    poly_modulus_degree: int
    coeff_mod_bit_sizes: list[int]
    global_scale_bits: int

    @property
    def slots(self) -> int:
        return self.poly_modulus_degree // 2

    @property
    def modulus_chain_bits(self) -> int:
        return sum(self.coeff_mod_bit_sizes)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def make_context(params: HEParams) -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=params.poly_modulus_degree,
        coeff_mod_bit_sizes=params.coeff_mod_bit_sizes,
    )
    context.global_scale = 2**params.global_scale_bits
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def time_call(fn):
    start = now_ms()
    value = fn()
    return value, now_ms() - start


def trimmed_mean(values: list[float]) -> float:
    if len(values) < 3:
        return float(np.mean(values))
    ordered = sorted(values)
    return float(np.mean(ordered[1:-1]))


def make_queries_and_cache(
    rng: np.random.Generator, trials: int, max_t: int, d_head: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = rng.normal(0, 0.35, size=(trials, d_head))
    keys = rng.normal(0, 0.35, size=(trials, max_t, d_head))
    values = rng.normal(0, 0.35, size=(trials, max_t, d_head))
    return queries, keys, values


def encrypted_token_scores(
    context: ts.Context, query: np.ndarray, key_cache: np.ndarray
) -> tuple[list[float], dict[str, float]]:
    encrypt_ms: list[float] = []
    compute_ms: list[float] = []
    decrypt_ms: list[float] = []
    serialize_ms: list[float] = []
    ciphertext_bytes = 0
    scores: list[float] = []

    q_enc, q_ms = time_call(lambda: ts.ckks_vector(context, query.tolist()))
    encrypt_ms.append(q_ms)
    q_bytes, q_ser_ms = time_call(lambda: len(q_enc.serialize()))
    serialize_ms.append(q_ser_ms)
    ciphertext_bytes += q_bytes

    for key in key_cache:
        k_enc, k_ms = time_call(lambda key=key: ts.ckks_vector(context, key.tolist()))
        encrypt_ms.append(k_ms)
        k_bytes, k_ser_ms = time_call(lambda k_enc=k_enc: len(k_enc.serialize()))
        serialize_ms.append(k_ser_ms)
        ciphertext_bytes += k_bytes

        score_ct, score_ms = time_call(lambda k_enc=k_enc: q_enc.dot(k_enc))
        compute_ms.append(score_ms)
        score, dec_ms = time_call(lambda score_ct=score_ct: score_ct.decrypt()[0])
        decrypt_ms.append(dec_ms)
        scores.append(float(score))

    return scores, {
        "encrypt_ms": sum(encrypt_ms),
        "compute_ms": sum(compute_ms),
        "decrypt_ms": sum(decrypt_ms),
        "serialize_ms": sum(serialize_ms),
        "ciphertext_bytes": float(ciphertext_bytes),
        "cache_ciphertexts": float(len(key_cache)),
        "query_ciphertexts": 1.0,
        "score_ciphertexts": float(len(key_cache)),
        "ct_ct_mul_ops": float(len(key_cache)),
        "slot_sum_ops": float(len(key_cache)),
        "mask_mul_ops": 0.0,
    }


def encrypted_block_scores(
    context: ts.Context, query: np.ndarray, key_cache: np.ndarray, block_size: int
) -> tuple[list[float], dict[str, float]]:
    encrypt_ms: list[float] = []
    compute_ms: list[float] = []
    decrypt_ms: list[float] = []
    serialize_ms: list[float] = []
    ciphertext_bytes = 0
    scores: list[float] = []
    d_head = len(query)

    num_blocks = math.ceil(len(key_cache) / block_size)
    q_blocks: list[np.ndarray] = []
    for block_id in range(num_blocks):
        start = block_id * block_size
        end = min(start + block_size, len(key_cache))
        actual_b = end - start
        q_blocks.append(np.tile(query, actual_b))

    for block_id, q_block in enumerate(q_blocks):
        start = block_id * block_size
        end = min(start + block_size, len(key_cache))
        key_block = key_cache[start:end].reshape(-1)

        q_enc, q_ms = time_call(lambda q_block=q_block: ts.ckks_vector(context, q_block.tolist()))
        k_enc, k_ms = time_call(lambda key_block=key_block: ts.ckks_vector(context, key_block.tolist()))
        encrypt_ms.extend([q_ms, k_ms])

        q_bytes, q_ser_ms = time_call(lambda q_enc=q_enc: len(q_enc.serialize()))
        k_bytes, k_ser_ms = time_call(lambda k_enc=k_enc: len(k_enc.serialize()))
        serialize_ms.extend([q_ser_ms, k_ser_ms])
        ciphertext_bytes += q_bytes + k_bytes

        products, mul_ms = time_call(lambda q_enc=q_enc, k_enc=k_enc: q_enc * k_enc)
        compute_ms.append(mul_ms)

        for local_idx in range(end - start):
            mask = np.zeros(len(key_block), dtype=np.float64)
            mask[local_idx * d_head : (local_idx + 1) * d_head] = 1.0
            score_ct, sum_ms = time_call(
                lambda products=products, mask=mask: (products * mask.tolist()).sum()
            )
            compute_ms.append(sum_ms)
            score, dec_ms = time_call(lambda score_ct=score_ct: score_ct.decrypt()[0])
            decrypt_ms.append(dec_ms)
            scores.append(float(score))

    return scores, {
        "encrypt_ms": sum(encrypt_ms),
        "compute_ms": sum(compute_ms),
        "decrypt_ms": sum(decrypt_ms),
        "serialize_ms": sum(serialize_ms),
        "ciphertext_bytes": float(ciphertext_bytes),
        "cache_ciphertexts": float(num_blocks),
        "query_ciphertexts": float(num_blocks),
        "score_ciphertexts": float(len(key_cache)),
        "ct_ct_mul_ops": float(num_blocks),
        "slot_sum_ops": float(len(key_cache)),
        "mask_mul_ops": float(len(key_cache)),
    }


def plain_attention(query: np.ndarray, keys: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    scores = keys @ query
    shifted = scores - scores.max()
    weights = np.exp(shifted) / np.exp(shifted).sum()
    context = weights @ values
    return {"scores": scores, "weights": weights, "context": context}


def polynomial_attention_from_scores(
    scores: np.ndarray, values: np.ndarray, degree: int
) -> np.ndarray:
    # HE-friendly positive polynomial proxy for exp(x), normalized in plaintext
    # here only to quantify model-side approximation error from the proposal.
    if degree == 2:
        positive = 1.0 + scores + 0.5 * scores**2
    elif degree == 3:
        positive = 1.0 + scores + 0.5 * scores**2 + (scores**3) / 6.0
    else:
        raise ValueError(f"unsupported degree: {degree}")
    positive = np.maximum(positive, 1e-9)
    weights = positive / positive.sum()
    return weights @ values


def run_accuracy_and_latency(
    context: ts.Context,
    params: HEParams,
    trials: int,
    context_lengths: list[int],
    block_sizes: list[int],
    d_head: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    max_t = max(context_lengths)
    queries, keys, values = make_queries_and_cache(rng, trials, max_t, d_head)
    rows: list[dict[str, Any]] = []
    approx_rows: list[dict[str, Any]] = []

    for t_len in context_lengths:
        for trial in range(trials):
            query = queries[trial]
            key_cache = keys[trial, :t_len]
            value_cache = values[trial, :t_len]
            plain = plain_attention(query, key_cache, value_cache)

            for degree in (2, 3):
                poly_context = polynomial_attention_from_scores(
                    plain["scores"], value_cache, degree
                )
                approx_rows.append(
                    {
                        "context_length": t_len,
                        "trial": trial,
                        "softmax_proxy_degree": degree,
                        "context_l2_error": float(
                            np.linalg.norm(poly_context - plain["context"])
                        ),
                        "context_linf_error": float(
                            np.max(np.abs(poly_context - plain["context"]))
                        ),
                    }
                )

            for block_size in block_sizes:
                if block_size == 1:
                    scores, metrics = encrypted_token_scores(context, query, key_cache)
                    packing = "token-wise"
                else:
                    scores, metrics = encrypted_block_scores(
                        context, query, key_cache, block_size
                    )
                    packing = "block-wise"

                score_array = np.array(scores)
                plain_scores = plain["scores"]
                rows.append(
                    {
                        "context_length": t_len,
                        "trial": trial,
                        "packing": packing,
                        "block_size": block_size,
                        "d_head": d_head,
                        "poly_modulus_degree": params.poly_modulus_degree,
                        "slots": params.slots,
                        "modulus_chain_bits": params.modulus_chain_bits,
                        "global_scale_bits": params.global_scale_bits,
                        "encrypt_ms": metrics["encrypt_ms"],
                        "compute_ms": metrics["compute_ms"],
                        "decrypt_ms": metrics["decrypt_ms"],
                        "serialize_ms": metrics["serialize_ms"],
                        "end_to_end_ms": metrics["encrypt_ms"]
                        + metrics["compute_ms"]
                        + metrics["decrypt_ms"],
                        "ciphertext_bytes": metrics["ciphertext_bytes"],
                        "cache_ciphertexts": metrics["cache_ciphertexts"],
                        "query_ciphertexts": metrics["query_ciphertexts"],
                        "score_ciphertexts": metrics["score_ciphertexts"],
                        "ct_ct_mul_ops": metrics["ct_ct_mul_ops"],
                        "slot_sum_ops": metrics["slot_sum_ops"],
                        "mask_mul_ops": metrics["mask_mul_ops"],
                        "score_l2_error": float(
                            np.linalg.norm(score_array - plain_scores)
                        ),
                        "score_linf_error": float(
                            np.max(np.abs(score_array - plain_scores))
                        ),
                        "score_mean_abs_error": float(
                            np.mean(np.abs(score_array - plain_scores))
                        ),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(approx_rows)


def summarize_latency(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["context_length", "packing", "block_size"], as_index=False)
    return grouped.agg(
        end_to_end_ms_mean=("end_to_end_ms", "mean"),
        end_to_end_ms_std=("end_to_end_ms", "std"),
        compute_ms_mean=("compute_ms", "mean"),
        encrypt_ms_mean=("encrypt_ms", "mean"),
        decrypt_ms_mean=("decrypt_ms", "mean"),
        ciphertext_mb_mean=("ciphertext_bytes", lambda x: float(np.mean(x) / 1_000_000)),
        cache_ciphertexts_mean=("cache_ciphertexts", "mean"),
        ct_ct_mul_ops_mean=("ct_ct_mul_ops", "mean"),
        slot_sum_ops_mean=("slot_sum_ops", "mean"),
        mask_mul_ops_mean=("mask_mul_ops", "mean"),
        score_l2_error_mean=("score_l2_error", "mean"),
        score_linf_error_mean=("score_linf_error", "mean"),
        score_mean_abs_error_mean=("score_mean_abs_error", "mean"),
    )


def bootstrap_placement_analysis(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # Calibrated by measured compute_ms per operation family. This is not
    # reported as actual bootstrapping execution because TenSEAL lacks it.
    rows: list[dict[str, Any]] = []
    bootstrap_ms = 30_000.0
    base_budget = 180.0
    min_budget = 40.0
    fixed_period_layers = 4
    layers = 12

    for _, row in summary.iterrows():
        context_len = int(row["context_length"])
        block_size = int(row["block_size"])
        packing = str(row["packing"])
        per_layer_compute_ms = float(row["compute_ms_mean"])
        token_blocks = math.ceil(context_len / block_size)
        layer_budget_cost = 6.0 + 0.55 * token_blocks + 0.15 * math.log2(context_len + 1)

        dynamic_bootstraps = 0
        budget = base_budget
        dynamic_trace = []
        for layer in range(1, layers + 1):
            if budget - layer_budget_cost < min_budget:
                dynamic_bootstraps += 1
                dynamic_trace.append(layer)
                budget = base_budget
            budget -= layer_budget_cost

        fixed_bootstraps = max(0, math.ceil(layers / fixed_period_layers) - 1)
        no_bootstrap_final_budget = base_budget - layers * layer_budget_cost
        dynamic_total_ms = layers * per_layer_compute_ms + dynamic_bootstraps * bootstrap_ms
        fixed_total_ms = layers * per_layer_compute_ms + fixed_bootstraps * bootstrap_ms

        rows.append(
            {
                "context_length": context_len,
                "packing": packing,
                "block_size": block_size,
                "layers": layers,
                "calibrated_layer_compute_ms": per_layer_compute_ms,
                "estimated_layer_budget_cost": layer_budget_cost,
                "no_bootstrap_final_budget": no_bootstrap_final_budget,
                "dynamic_bootstraps": dynamic_bootstraps,
                "dynamic_bootstrap_layers": ",".join(map(str, dynamic_trace)) or "none",
                "fixed_bootstraps": fixed_bootstraps,
                "dynamic_total_ms": dynamic_total_ms,
                "fixed_total_ms": fixed_total_ms,
                "dynamic_vs_fixed_speedup": fixed_total_ms / dynamic_total_ms
                if dynamic_total_ms > 0
                else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bootstrap_placement_calibrated.csv", index=False)
    return df


def make_plots(summary: pd.DataFrame, approx: pd.DataFrame, placement: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    label = lambda p, b: f"{p}, b={b}"
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for (packing, block_size), part in summary.groupby(["packing", "block_size"]):
        part = part.sort_values("context_length")
        ax.plot(
            part["context_length"],
            part["compute_ms_mean"],
            marker="o",
            label=label(packing, block_size),
        )
    ax.set_title("Real CKKS encrypted QK score computation")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Compute latency per token step (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_by_context.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for (packing, block_size), part in summary.groupby(["packing", "block_size"]):
        part = part.sort_values("context_length")
        ax.plot(
            part["context_length"],
            part["ciphertext_mb_mean"],
            marker="o",
            label=label(packing, block_size),
        )
    ax.set_title("Serialized ciphertext footprint")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Serialized ciphertexts per trial (MB)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ciphertext_footprint.png", dpi=180)
    plt.close(fig)

    approx_summary = approx.groupby(["context_length", "softmax_proxy_degree"], as_index=False).agg(
        context_l2_error_mean=("context_l2_error", "mean"),
        context_l2_error_std=("context_l2_error", "std"),
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for degree, part in approx_summary.groupby("softmax_proxy_degree"):
        part = part.sort_values("context_length")
        ax.plot(
            part["context_length"],
            part["context_l2_error_mean"],
            marker="s",
            label=f"degree {degree}",
        )
    ax.set_title("Softmax polynomial proxy error")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Context vector L2 error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "softmax_proxy_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for (packing, block_size), part in placement.groupby(["packing", "block_size"]):
        part = part.sort_values("context_length")
        ax.plot(
            part["context_length"],
            part["dynamic_bootstraps"],
            marker="o",
            label=label(packing, block_size),
        )
    ax.set_title("Dynamic bootstrap placement count (calibrated analysis)")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Estimated bootstraps over 12 layers")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bootstrap_placement.png", dpi=180)
    plt.close(fig)


def write_report(
    params: HEParams,
    summary: pd.DataFrame,
    approx: pd.DataFrame,
    placement: pd.DataFrame,
    out_dir: Path,
    elapsed_s: float,
):
    best_rows = []
    for t_len, part in summary.groupby("context_length"):
        token = part[part["block_size"] == 1].iloc[0]
        fastest = part.sort_values("compute_ms_mean").iloc[0]
        smallest = part.sort_values("ciphertext_mb_mean").iloc[0]
        best_rows.append(
            {
                "T": int(t_len),
                "token_compute_ms": float(token["compute_ms_mean"]),
                "fastest": f"{fastest['packing']} b={int(fastest['block_size'])}",
                "fastest_compute_ms": float(fastest["compute_ms_mean"]),
                "smallest": f"{smallest['packing']} b={int(smallest['block_size'])}",
                "smallest_ciphertext_mb": float(smallest["ciphertext_mb_mean"]),
            }
        )

    approx_summary = approx.groupby("softmax_proxy_degree", as_index=False).agg(
        mean_l2=("context_l2_error", "mean"),
        mean_linf=("context_linf_error", "mean"),
    )

    lines = [
        "# CKKS KV-cache-aware Packing Experiment Report",
        "",
        "## What was executed",
        "",
        "- Library: TenSEAL 0.3.16, backed by Microsoft SEAL.",
        "- Real HE operations: CKKS context generation, Galois/relinearization keys, encryption, ciphertext-ciphertext multiplication, plaintext masking, slot summation, decryption, and serialization.",
        "- Not executed: CKKS bootstrapping, because TenSEAL 0.3.x does not expose a bootstrapping API. The placement table is a calibrated scheduling analysis, not a fake bootstrap run.",
        "",
        "## Parameters",
        "",
        f"- Python: {PYTHON}",
        f"- Platform: {platform.platform()}",
        f"- Random seed: {SEED}",
        f"- poly_modulus_degree: {params.poly_modulus_degree}",
        f"- CKKS slots: {params.slots}",
        f"- coeff_mod_bit_sizes: {params.coeff_mod_bit_sizes}",
        f"- global_scale: 2^{params.global_scale_bits}",
        f"- Wall-clock runtime: {elapsed_s:.2f} s",
        "",
        "## Main findings",
        "",
    ]

    for row in best_rows:
        lines.append(
            f"- T={row['T']}: token-wise compute {row['token_compute_ms']:.2f} ms; "
            f"fastest measured variant was {row['fastest']} at "
            f"{row['fastest_compute_ms']:.2f} ms; smallest footprint was "
            f"{row['smallest']} at {row['smallest_ciphertext_mb']:.2f} MB."
        )

    lines.extend(
        [
            "",
            "## Softmax proxy error",
            "",
        ]
    )
    for _, row in approx_summary.iterrows():
        lines.append(
            f"- Degree {int(row['softmax_proxy_degree'])}: mean context L2 error "
            f"{row['mean_l2']:.6f}, mean Linf error {row['mean_linf']:.6f}."
        )

    dynamic_wins = placement[placement["dynamic_vs_fixed_speedup"] > 1.0]
    lines.extend(
        [
            "",
            "## Dynamic bootstrap placement analysis",
            "",
            f"- Dynamic placement is faster than the fixed 4-layer schedule in {len(dynamic_wins)}/{len(placement)} calibrated cases.",
            "- These numbers are useful for choosing placement policies, but they should be replaced by OpenFHE/SEAL bootstrapping measurements if an implementation with CKKS bootstrap support is added.",
            "",
            "## Generated artifacts",
            "",
            "- `raw_ckks_runs.csv`: every measured trial.",
            "- `summary_by_packing.csv`: grouped latency, storage, operation counts, and CKKS score error.",
            "- `softmax_proxy_error.csv`: polynomial Softmax approximation error from the same real score distributions.",
            "- `bootstrap_placement_calibrated.csv`: placement-policy analysis calibrated by measured CKKS operation costs.",
            "- `latency_by_context.png`, `ciphertext_footprint.png`, `softmax_proxy_error.png`, `bootstrap_placement.png`: plots.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--context-lengths", default="4,8,16,32")
    parser.add_argument("--block-sizes", default="1,2,4,8")
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--poly-modulus-degree", type=int, default=8192)
    parser.add_argument("--coeff-mod-bit-sizes", default="60,40,40,60")
    parser.add_argument("--global-scale-bits", type=int, default=40)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller sweep for fast smoke testing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    if args.quick:
        args.trials = 1
        context_lengths = context_lengths[:2]
        block_sizes = block_sizes[:2]
    params = HEParams(
        poly_modulus_degree=args.poly_modulus_degree,
        coeff_mod_bit_sizes=[int(x) for x in args.coeff_mod_bit_sizes.split(",")],
        global_scale_bits=args.global_scale_bits,
    )

    if max(block_sizes) * args.d_head > params.slots:
        raise ValueError("max block_size * d_head exceeds CKKS slot count")

    start = time.perf_counter()
    context = make_context(params)
    raw, approx = run_accuracy_and_latency(
        context,
        params,
        args.trials,
        context_lengths,
        block_sizes,
        args.d_head,
    )
    summary = summarize_latency(raw)
    placement = bootstrap_placement_analysis(summary, out_dir)
    elapsed_s = time.perf_counter() - start

    raw.to_csv(out_dir / "raw_ckks_runs.csv", index=False)
    summary.to_csv(out_dir / "summary_by_packing.csv", index=False)
    approx.to_csv(out_dir / "softmax_proxy_error.csv", index=False)
    make_plots(summary, approx, placement, out_dir)
    write_report(params, summary, approx, placement, out_dir, elapsed_s)

    metadata = {
        "python": PYTHON,
        "platform": platform.platform(),
        "seed": SEED,
        "params": params.__dict__,
        "trials": args.trials,
        "context_lengths": context_lengths,
        "block_sizes": block_sizes,
        "d_head": args.d_head,
        "elapsed_s": elapsed_s,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote results to {out_dir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
