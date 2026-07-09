# ---------------------------------------------------------------------------
# Build + profiling harness for the cuda-mma kernels.
# ---------------------------------------------------------------------------

NVCC      := nvcc
CXX_FLAGS := -std=c++17 -O3

# sm_80 = Ampere (A100 / RTX 30xx)   sm_86 = RTX 30xx consumer
# sm_70 = Volta (V100)                sm_75 = Turing (RTX 20xx / T4)
# sm_89 = Ada (RTX 40xx)
# Override on the command line: make ARCH=sm_75
ARCH      ?= sm_80

CUTLASS_DIR := vendor/cutlass

# --generate-line-info correlates ncu's source/SASS views back to this code.
NVCC_FLAGS := $(CXX_FLAGS) -arch=$(ARCH) \
              -rdc=true \
              --generate-line-info \
              -Xcompiler -Wall \
              -diag-suppress 20013,20015 \
              -I. -Iinclude -I$(CUTLASS_DIR)/include

BIN_DIR    := build/bin
TARGET     := $(BIN_DIR)/cuda_profiling
SRCS       := main.cu src/kernels.cu src/utils.cu src/profile_kernels.cu src/profile_cutlass.cu
HDRS       := include/cuda_check.cuh include/kernels.cuh include/timer.h \
              include/utils.h include/profiling.h

REPORT_DIR := reports

# Square matrix size used for profiling runs.
SIZE   ?= 4096
# One name from PROFILE_TABLE (see include/profiling.h): naive, coalesced,
# tiled, coarsened, cublas, cutlass_fp32, cutlass_tf32, cutlass_fp16.
# Required for `make ncu`; if empty, `make run`/`make nsys` profile all of them.
KERNEL ?=

.PHONY: all clean run vendor ncu ncu-metrics nsys nsys-stats

all: $(TARGET)

$(BIN_DIR) $(REPORT_DIR):
	mkdir -p $@

$(TARGET): $(SRCS) $(HDRS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRCS) -lcublas -ldl \
	    -Xlinker -rpath,/usr/local/cuda/targets/x86_64-linux/lib

vendor:
	./scripts/download_vendor.sh

run: $(TARGET)
	./$(TARGET) $(SIZE) $(KERNEL)

# --- Nsight Compute --------------------------------------------------------
# Deep-dive a single kernel (occupancy, roofline, memory workload, etc).
# The NVTX range and the binary's own KERNEL filter both narrow execution to
# just that kernel, so `--set full`'s replay passes stay fast.
#   make ncu KERNEL=tiled SIZE=2048
ncu: $(TARGET) | $(REPORT_DIR)
	@if [ -z "$(KERNEL)" ]; then \
	    echo "Usage: make ncu KERNEL=<name> [SIZE=4096]"; \
	    echo "Available: naive coalesced tiled coarsened cublas cutlass_fp32 cutlass_tf32 cutlass_fp16"; \
	    exit 1; \
	fi
	ncu --set full \
	    --nvtx --nvtx-include "$(KERNEL)/" \
	    --launch-count 1 \
	    -f -o $(REPORT_DIR)/ncu_$(KERNEL)_$(SIZE) \
	    ./$(TARGET) $(SIZE) $(KERNEL)

# Fast, fixed metric set across every kernel in one pass (CSV on stdout).
# Same metrics as cuda-mma's `make profile`, plus an SM throughput summary.
ncu-metrics: $(TARGET) | $(REPORT_DIR)
	ncu --metrics \
	    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
	    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
	    sm__warps_active.avg.pct_of_peak_sustained_active,\
	    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
	    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
	    --nvtx --csv \
	    ./$(TARGET) $(SIZE) $(KERNEL) > $(REPORT_DIR)/ncu_metrics_$(SIZE).csv

# --- Nsight Systems ---------------------------------------------------------
# Whole-application timeline: every kernel launch, memcpy and NVTX range.
# Open reports/nsys_<SIZE>.nsys-rep in the Nsight Systems UI to inspect it.
nsys: $(TARGET) | $(REPORT_DIR)
	nsys profile \
	    --trace=cuda,nvtx,osrt \
	    --force-overwrite=true \
	    -o $(REPORT_DIR)/nsys_$(SIZE) \
	    ./$(TARGET) $(SIZE) $(KERNEL)

# Per-kernel duration summary extracted from the most recent `make nsys` run.
nsys-stats: | $(REPORT_DIR)
	nsys stats --report cuda_gpu_kern_sum --format csv \
	    --output $(REPORT_DIR)/nsys_$(SIZE)_kernels \
	    $(REPORT_DIR)/nsys_$(SIZE).nsys-rep

clean:
	rm -rf build reports
