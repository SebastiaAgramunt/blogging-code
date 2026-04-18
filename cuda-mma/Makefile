# ---------------------------------------------------------------------------
# Build system for cuda-mma
# ---------------------------------------------------------------------------

NVCC      := nvcc
CXX_FLAGS := -std=c++17 -O3

# sm_80 = Ampere (A100 / RTX 30xx)   sm_86 = RTX 30xx consumer
# sm_70 = Volta (V100)                sm_75 = Turing (RTX 20xx / T4)
# sm_89 = Ada (RTX 40xx)
# Override on the command line: make ARCH=sm_75
ARCH      ?= sm_80

NVCC_FLAGS := $(CXX_FLAGS) -arch=$(ARCH) \
              -rdc=true \
              --generate-line-info \
              -Xcompiler -Wall \
              -I. -Iinclude

BIN_DIR   := build/bin
TARGET    := $(BIN_DIR)/cuda_mma
TEST      := $(BIN_DIR)/test_correctness
SRCS      := main.cu src/kernels.cu src/utils.cu src/benchmarks.cu
TEST_SRCS := test/test_correctness.cu src/kernels.cu src/utils.cu
HDRS      := include/cuda_check.cuh include/kernels.cuh \
             include/timer.h include/utils.h include/benchmarks.h

.PHONY: all clean run test profile

all: $(TARGET) $(TEST)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRCS) $(HDRS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRCS)

$(TEST): $(TEST_SRCS) $(HDRS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(TEST_SRCS)

run: $(TARGET)
	./$(TARGET)

test: $(TEST)
	./$(TEST)

# Requires Nsight Compute (ncu).
profile: $(TARGET)
	ncu --metrics \
	    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
	    l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
	    sm__warps_active.avg.pct_of_peak_sustained_active,\
	    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
	    ./$(TARGET)

clean:
	rm -rf build
