# Build Python bindings with Pybind11

Supporting material for blogpost [agramunt.me/posts/cpp-python-extension/](https://agramunt.me/posts/cpp-python-extension/).

## Instructions

Compile python library running

```bash
# recreate build directory
TASK=clean ./scripts/compile.sh

# compile library
TASK=compile_python_lib ./scripts/compile.sh

# run a simple test
TASK=test ./scripts/compile.sh
```
