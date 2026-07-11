# Profiling the cuda-mma SGEMM kernels with Nsight Compute and Nsight Systems

Same SGEMM kernels as [cuda-mma](../cuda-mma) (naive, coalesced, tiled,
coarsened, cuBLAS, CUTLASS fp32/tf32/fp16), restructured for profiling
instead of roofline benchmarking: one binary, one NVTX range per kernel,
no CSV sweep.

## Build

```bash
./scripts/download_vendor.sh   # fetches CUTLASS headers into vendor/
make                            # builds build/bin/cuda_profiling
```

## Run directly

```bash
./build/bin/cuda_profiling [SIZE] [KERNEL]
```

`SIZE` is the square matrix dimension (default 4096). `KERNEL` restricts
the run to one implementation — `naive`, `coalesced`, `tiled`,
`coarsened`, `cublas`, `cutlass_fp32`, `cutlass_tf32`, or `cutlass_fp16`
(see `include/profiling.h`). Omit it to run all of them in sequence.

Every kernel's profiled launches are wrapped in an `NvtxRange` named after
it (`include/profiling.h`). That serves two purposes: Nsight Systems shows
it as a labeled block on the timeline, and Nsight Compute can be told to
collect metrics only inside that block with `--nvtx-include`.

## Nsight Systems — whole-application timeline

```bash
make nsys SIZE=4096                 # -> reports/nsys_4096.nsys-rep
make nsys-stats SIZE=4096           # -> reports/nsys_4096_kernels.csv
```

Open the `.nsys-rep` in the Nsight Systems UI to see every kernel launch,
memcpy, and NVTX range on one timeline — useful for spotting gaps between
launches, H2D/D2H transfer overlap, and the relative wall-clock cost of
each implementation back-to-back.

## Nsight Compute — per-kernel deep dive

```bash
make ncu KERNEL=tiled SIZE=2048     # -> reports/ncu_tiled_2048.ncu-rep
```

This runs `--set full` scoped to the `tiled` NVTX range only
(`--nvtx-include "tiled/"`) and limited to the first launch
(`--launch-count 1`), so the replay passes a full section set requires
stay fast even though the binary itself launches every kernel after it
in the warmup loop. `KERNEL` is required for this target — `make ncu`
on its own prints the list of valid names.

For a fast metrics-only sweep across every kernel in one pass (occupancy,
memory throughput) instead of one deep dive:

```bash
make ncu-metrics SIZE=4096          # -> reports/ncu_metrics_4096.csv
```

## Notes

- `--generate-line-info` is on by default in the `Makefile` so Nsight
  Compute's source/SASS view can map hotspots back to this code.
- `KERNEL`/`SIZE` are plain Makefile variables — override on the command
  line as shown above, or `export` them before calling `make run`.
- `make clean` removes `build/` and `reports/`; `vendor/` (CUTLASS) is
  left alone since it's only fetched once.
