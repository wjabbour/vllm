# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure real TTFT/TPOT for Llama-3.1-8B-Instruct-FP8 on RDNA4 (gfx1201)
using the async streaming engine directly (bypasses the broken xgrammar
import in the full server CLI on this checkout).
"""

import asyncio
import time

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

MODEL = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main causes of the French Revolution?",
    "Describe how a transformer neural network works.",
    "Summarize the plot of Romeo and Juliet.",
    "What is the difference between TCP and UDP?",
    "Give me a recipe for a classic margherita pizza.",
    "Explain quantum entanglement to a curious teenager.",
] * 2  # 16 concurrent requests, same as the profiling run

SAMPLING = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)


async def run_one(engine, prompt, req_id):
    start = time.perf_counter()
    first_token_time = None
    token_times = []
    async for out in engine.generate(prompt, SAMPLING, request_id=req_id):
        now = time.perf_counter()
        if first_token_time is None and out.outputs[0].token_ids:
            first_token_time = now
        token_times.append((now, len(out.outputs[0].token_ids)))
    end = time.perf_counter()
    n_tokens = token_times[-1][1] if token_times else 0
    ttft = (first_token_time - start) if first_token_time else None
    # TPOT = time from first token to last token / (n_tokens - 1)
    tpot = (
        (end - first_token_time) / (n_tokens - 1)
        if first_token_time and n_tokens > 1
        else None
    )
    return ttft, tpot, n_tokens, end - start


async def main():
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=MODEL,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            max_model_len=4096,
        )
    )

    # Warmup (outside measurement).
    async for _ in engine.generate(
        "Hello", SamplingParams(temperature=0.0, max_tokens=16), request_id="warmup"
    ):
        pass

    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[run_one(engine, p, f"req-{i}") for i, p in enumerate(PROMPTS)]
    )
    wall = time.perf_counter() - t0

    ttfts = [r[0] for r in results if r[0] is not None]
    tpots = [r[1] for r in results if r[1] is not None]
    total_tokens = sum(r[2] for r in results)

    print("-" * 60)
    print(f"Concurrent requests: {len(PROMPTS)}")
    print(f"Wall time: {wall:.3f}s")
    print(f"Total output tokens: {total_tokens}")
    print(f"Aggregate output throughput: {total_tokens / wall:.1f} tok/s")
    print("-" * 60)
    ttfts.sort()
    tpots.sort()

    def pct(xs, p):
        if not xs:
            return float("nan")
        idx = min(int(len(xs) * p), len(xs) - 1)
        return xs[idx]

    print(
        f"TTFT   mean={sum(ttfts) / len(ttfts) * 1e3:.1f}ms  "
        f"p50={pct(ttfts, 0.5) * 1e3:.1f}ms  p90={pct(ttfts, 0.9) * 1e3:.1f}ms  "
        f"p99={pct(ttfts, 0.99) * 1e3:.1f}ms"
    )
    print(
        f"TPOT   mean={sum(tpots) / len(tpots) * 1e3:.2f}ms  "
        f"p50={pct(tpots, 0.5) * 1e3:.2f}ms  p90={pct(tpots, 0.9) * 1e3:.2f}ms  "
        f"p99={pct(tpots, 0.99) * 1e3:.2f}ms"
    )
    print("-" * 60)

    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
