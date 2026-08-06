# Define build directories
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin
LIB_DIR := $(BUILD_DIR)/lib

# Include and library paths
INCLUDES := -Iexternal/lib/opencv/include/opencv4 \
            -Iinclude \
            -I/usr/local/cuda/include

LIBS := -Lexternal/lib/opencv/lib \
        -L/usr/local/cuda/lib64 \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
        -lopencv_imgcodecs \
        -lcudart \
        -Wl,-rpath,external/lib/opencv/lib

NVCC_FLAGS := -std=c++17 -Xcudafe --diag_suppress=611 $(INCLUDES)
CXX := g++
NVCC := nvcc

# Source files
SRCS := src/main.cu \
        src/utils.cpp \
        src/rgb_to_grayscale.cu \
        src/blur_image.cu

# Object files
OBJS := $(patsubst src/%.cu,$(OBJ_DIR)/%.o,$(filter %.cu,$(SRCS))) \
        $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(filter %.cpp,$(SRCS)))

# Output binary
TARGET := $(BIN_DIR)/main

# Default target
all: $(TARGET)

# Create build directories
$(OBJ_DIR) $(BIN_DIR) $(LIB_DIR):
	@mkdir -p $@

# Compile object files
$(OBJ_DIR)/%.o: src/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Link the final executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $^ $(INCLUDES) $(LIBS) -o $@

# Cleanup
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

