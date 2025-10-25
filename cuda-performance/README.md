# CUDA performance on vector addition and matrix multiplication

Supporting material for blogpost [agramunt.me/posts/cuda-performance/](https://agramunt.me/posts/cuda-performance/).

## Compile & Run

Compile the code (all intermediate objects will be placed in `build` directory)

```bash
./scripts/compile.sh
```

Execute the benchmark

```bash
./scripts/execute.sh
```

## Analyze data and generate plots

First create a new python environment and install depencencies, you can do that with

```bash
./scripts/install_python_env.sh
```

That will create an environment in the root directory `.venv`. Then just execute the script with

```bash
.venv/bin/python scripts/analyze.py
```
