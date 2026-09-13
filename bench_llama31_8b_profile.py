# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Profile Llama-3.1-8B-Instruct-FP8 decode/prefill on RDNA4 (gfx1201) to find
hot kernels worth tuning. Produces a torch profiler trace under ./vllm_profile.
"""

import time

from vllm import LLM, SamplingParams

MODEL = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"

WARMUP_PROMPTS = ["The quick brown fox jumps over the lazy dog."] * 4

# A batch mixing short and long prompts so the trace captures both prefill
# and a meaningful number of decode steps.
PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main causes of the French Revolution?",
    "Describe how a transformer neural network works.",
    "Summarize the plot of Romeo and Juliet.",
    "What is the difference between TCP and UDP?",
    "Give me a recipe for a classic margherita pizza.",
    "Explain quantum entanglement to a curious teenager.",
] * 2  # 16 concurrent sequences

SAMPLING = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        max_model_len=4096,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "./vllm_profile",
        },
    )

    # Warm up: trigger kernel compilation / CUDA graph capture outside the trace.
    llm.generate(WARMUP_PROMPTS, SamplingParams(temperature=0.0, max_tokens=32))

    llm.start_profile()
    outputs = llm.generate(PROMPTS, SAMPLING)
    llm.stop_profile()

    for out in outputs[:2]:
        print("-" * 50)
        print("Prompt:", out.prompt)
        print("Output:", out.outputs[0].text[:200])

    # Let the profiler flush trace files (worker runs in a subprocess).
    time.sleep(10)
    print("Trace written to ./vllm_profile")


if __name__ == "__main__":
    main()
