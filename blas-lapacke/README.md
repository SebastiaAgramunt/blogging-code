# BLAS and LAPACK install

Supporting material for blogpost [https://agramunt.me/posts/blas-lapack/](https://agramunt.me/posts/blas-lapack/).

## Instructions

Compile python library running

```bash
# install blas and lapacke
./scripts/install-external-libraries.sh

# build examples and run
./scripts/build-run.sh

# execute binaries
./build/bin/cblas_example
./build/bin/lapacke_example
```
