# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compare the ROCm fp8 skinny decode GEMM `wvSplitKQ` against AITER / hipBLASLt
fp8 paths on decode-shaped per-tensor W8A8 problems.

Providers (per shape + batch):
    torch-scaled-mm   torch._scaled_mm            (hipBLASLt fp8)
    wvSplitKQ         ops.wvSplitKQ               (csrc/rocm/skinny_gemms.cu)
    aiter-a8w8-ck     aiter.gemm_a8w8_CK

Mirrors the dispatch in
vllm/model_executor/kernels/linear/scaled_mm/rocm.py: wvSplitKQ is used when
`M in (1, 2, 3, 4, 8, 16)` and `N % 16 == 0` and `K % 16 == 0`, else
torch._scaled_mm.

Usage:
    python benchmark_rocm_wvsplitkq_vs_aiter_fp8.py --metric latency
"""

import argparse
import copy
import itertools
import os

import torch
from benchmark_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units

FP8 = (
    current_platform.fp8_dtype() if current_platform.is_rocm() else torch.float8_e4m3fn
)


def per_tensor_quant_fp8(x: torch.Tensor):
    fmax = torch.finfo(FP8).max
    amax = x.abs().max().clamp(min=1e-12)
    scale = (amax / fmax).float().view(1)
    xq = (x / scale).clamp(-fmax, fmax).to(FP8)
    return xq, scale


def _load_aiter_ck():
    try:
        from aiter import gemm_a8w8_CK

        return gemm_a8w8_CK
    except Exception:
        return None


def _dispatch(provider, Aq, Wq, As, Bs, bias, out_dtype, cu_count):
    M = Aq.shape[0]
    N, K = Wq.shape
    if provider == "torch-scaled-mm":
        return torch._scaled_mm(
            Aq, Wq.t(), out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )
    if provider == "wvSplitKQ":
        if M not in (1, 2, 3, 4, 8, 16) or N % 16 or K % 16:
            return None
        return ops.wvSplitKQ(Wq, Aq, out_dtype, As, Bs, cu_count, bias)
    if provider == "aiter-a8w8-ck":
        ck = _load_aiter_ck()
        if ck is None:
            return None
        out = ck(Aq, Wq, As, Bs, None, out_dtype)
        if bias is not None:
            out = out + bias
        return out
    return None


def check_provider(provider, Aq, Wq, As, Bs, bias, out_dtype, cu_count, ref):
    try:
        out = _dispatch(provider, Aq, Wq, As, Bs, bias, out_dtype, cu_count)
    except Exception as e:  # noqa: BLE001
        return f"ERROR ({type(e).__name__}: {e})"
    if out is None:
        return "n/a"
    out = out[: Aq.shape[0]].to(torch.float32)
    max_abs = (out - ref).abs().max().item()
    denom = ref.abs().max().item() or 1.0
    ok = max_abs / denom < 5e-2
    return f"{'PASS' if ok else 'FAIL'} (max_rel={max_abs / denom:.2e})"


def build_benchmark(providers, batch_sizes, metric):
    ylabel = (
        "latency (us, smaller is better)"
        if metric == "latency"
        else "TFLOP/s (larger is better)"
    )

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=list(batch_sizes),
            x_log=False,
            line_arg="provider",
            line_vals=providers,
            line_names=providers,
            ylabel=ylabel,
            plot_name="rocm-wvsplitkq-vs-aiter-fp8",
            args={},
        )
    )
    def benchmark(batch_size, provider, Wq, Bs, bias, out_dtype, metric):
        M = batch_size
        N, K = Wq.shape
        device = Wq.device
        Af = torch.randn((M, K), device=device, dtype=out_dtype) * (2.0 / K) ** 0.5
        Aq, As = per_tensor_quant_fp8(Af)
        cu_count = num_compute_units()

        try:
            probe = _dispatch(provider, Aq, Wq, As, Bs, bias, out_dtype, cu_count)
        except Exception:  # noqa: BLE001
            probe = None
        if probe is None:
            return (float("nan"),) * 3

        quantiles = [0.5, 0.2, 0.8]
        bench = (
            triton.testing.do_bench
            if os.environ.get("SKINNY_BENCH_NOGRAPH")
            else triton.testing.do_bench_cudagraph
        )
        ms, min_ms, max_ms = bench(
            lambda: _dispatch(provider, Aq, Wq, As, Bs, bias, out_dtype, cu_count),
            quantiles=quantiles,
        )
        if metric == "latency":
            return ms * 1e3, max_ms * 1e3, min_ms * 1e3
        tflops = lambda t: (2 * M * N * K) * 1e-12 / (t * 1e-3)
        return tflops(ms), tflops(max_ms), tflops(min_ms)

    return benchmark


def prepare_shapes(models, tp_sizes):
    out = []
    for model, tp in itertools.product(models, tp_sizes):
        for K, N in copy.deepcopy(WEIGHT_SHAPES[model]):
            if N >= K:
                N //= tp
            else:
                K //= tp
            out.append((model, N, K))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="ROCm wvSplitKQ vs AITER fp8 benchmark"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/Llama-2-70b-hf/TP1"],
        choices=list(WEIGHT_SHAPES.keys()),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 3, 4, 8, 16]
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--metric", choices=["tflops", "latency"], default="latency")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument(
        "--providers",
        type=str,
        default=None,
        help="comma-separated subset of torch-scaled-mm,wvSplitKQ,aiter-a8w8-ck",
    )
    parser.add_argument("--save-path", type=str, default="wvsplitkq_vs_aiter")
    args = parser.parse_args()

    if not current_platform.is_rocm():
        raise SystemExit("ROCm only.")

    out_dtype = torch.bfloat16
    if args.providers:
        providers = args.providers.split(",")
    else:
        providers = ["torch-scaled-mm", "wvSplitKQ"]
        if _load_aiter_ck() is not None:
            providers.append("aiter-a8w8-ck")
    print(f"providers: {providers}  fp8={FP8}  cu={num_compute_units()}\n")

    benchmark = build_benchmark(providers, args.batch_sizes, args.metric)

    for model, N, K in prepare_shapes(args.models, args.tp_sizes):
        Wf = torch.randn((N, K), device="cuda", dtype=out_dtype) * (2.0 / K) ** 0.5
        Wq, Bs = per_tensor_quant_fp8(Wf)
        bias = (
            torch.randn(N, device="cuda", dtype=out_dtype) * 0.01 if args.bias else None
        )
        header = f"{model}  N={N} K={K}"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")

        if not args.no_check:
            cu_count = num_compute_units()
            for bs in args.batch_sizes:
                Af = (
                    torch.randn((bs, K), device="cuda", dtype=out_dtype)
                    * (2.0 / K) ** 0.5
                )
                Aq, As = per_tensor_quant_fp8(Af)
                ref = (Aq.float() * As) @ (Wq.float() * Bs).t() + (
                    bias.float() if bias is not None else 0.0
                )
                results = []
                for p in providers:
                    r = check_provider(
                        p, Aq, Wq, As, Bs, bias, out_dtype, cu_count, ref
                    )
                    results.append(f"{p}: {r}")
                print(f"  bs={bs:<4} " + "  ".join(results))

        save_path = os.path.join(args.save_path, f"n{N}_k{K}")
        os.makedirs(save_path, exist_ok=True)
        benchmark.run(
            print_data=True,
            show_plots=False,
            save_path=save_path,
            Wq=Wq,
            Bs=Bs,
            bias=bias,
            out_dtype=out_dtype,
            metric=args.metric,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
