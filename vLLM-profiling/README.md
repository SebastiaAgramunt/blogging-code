# vLLM Profiling

Profiles a vLLM batch-inference run with Nsight Systems to inspect CUDA kernel activity, NVTX ranges, and OS-level events during generation.

## What it does

`bench.py` loads `meta-llama/Meta-Llama-3-8B-Instruct` with vLLM and runs a batch of 32 prompts (4 topics x 8 repeats) through `llm.generate`. `benchmark.sh` wraps that script with `nsys profile`, capturing a trace you can open in the Nsight Systems GUI.

## Files

- `install.sh` — creates a `.venv`, installs `vllm` + `huggingface_hub`, and checks the GPU driver via `nvidia-smi`. Nsight Systems CLI installation is included but commented out (assumes it's already installed at `/opt/nvidia/nsight-systems-cli`).
- `benchmark.sh` — activates the venv, loads `.env`, and runs `bench.py` under `nsys profile` (trace: `cuda,nvtx,osrt`, with CUDA graph node tracing).
- `bench.py` — the actual vLLM workload being profiled.
- `.env` — local secrets (gitignored). Must define `HF_TOKEN` for the gated Llama 3 model.
- `*.nsys-rep` — Nsight Systems trace output (gitignored).

## Prerequisites

- NVIDIA GPU + driver (verified by `nvidia-smi` in `install.sh`)
- Nsight Systems CLI (`nsys`) installed and on `PATH`, or available at `/opt/nvidia/nsight-systems-cli`
- A Hugging Face token with access to `meta-llama/Meta-Llama-3-8B-Instruct`

## Usage

```bash
# one-time setup
./install.sh

# create .env with your HF token
echo 'export HF_TOKEN=hf_...' > .env

# run the profiled benchmark
./benchmark.sh
```

This produces `vllm_llama3_8b_bs32.nsys-rep`, which can be opened with `nsys-ui` or the Nsight Systems desktop app.
