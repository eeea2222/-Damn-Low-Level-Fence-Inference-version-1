# =============================================================================
# Makefile — Fence Inference Engine
# Target: NVIDIA RTX 4060 (Ada Lovelace, SM 8.9)
#
# Usage:
#   make fence                      build the main binary
#   make test_q6k_dot               run the Q6_K dot-product kernel test
#   make test_gguf_parser MODEL=... run the GGUF parser test against a model
#   make test_gemv_q6k MODEL=...    run the GEMV Q6_K kernel test
#   make debug_forward MODEL=...    build the layer-stats debug tool
#   make clean                      remove build artefacts
# =============================================================================

NVCC        := nvcc
CXX         := g++

BUILD_DIR   := build
SRC_DIR     := src
TEST_DIR    := tests
KERNEL_DIR  := $(SRC_DIR)/kernels
GGUF_DIR    := $(SRC_DIR)/gguf
MODEL_DIR   := $(SRC_DIR)/model
TOK_DIR     := $(SRC_DIR)/tokenizer

SM          := 89
CUDA_ARCH   := -gencode arch=compute_$(SM),code=sm_$(SM)

NVCC_FLAGS  := $(CUDA_ARCH) \
               -std=c++17 \
               -O3 \
               --use_fast_math \
               -Xcompiler -Wall \
               -I$(SRC_DIR)

CXX_FLAGS   := -std=c++17 -O3 -Wall -I$(SRC_DIR)

ifdef DEBUG
    NVCC_FLAGS += -g -G -DDEBUG -lineinfo
    CXX_FLAGS  += -g -DDEBUG
endif

.PHONY: all clean fence \
        test_q6k_dot test_gguf_parser test_gemv_q6k test debug_forward

all: fence

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ---- Shared objects ----
$(BUILD_DIR)/gguf_parser.o: $(GGUF_DIR)/gguf_parser.cpp $(GGUF_DIR)/gguf_parser.h | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

$(BUILD_DIR)/tokenizer.o: $(TOK_DIR)/tokenizer.cpp $(TOK_DIR)/tokenizer.h \
                          $(MODEL_DIR)/chat_template.h | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c -o $@ $<

# ---- Main inference engine ----
FENCE_DEPS := $(SRC_DIR)/main.cu \
              $(MODEL_DIR)/qwen3.h $(MODEL_DIR)/qwen3.cu \
              $(MODEL_DIR)/chat_template.h \
              $(KERNEL_DIR)/q6k_types.h $(KERNEL_DIR)/q6k_dot.cuh $(KERNEL_DIR)/gemv_q6k.cuh \
              $(KERNEL_DIR)/q8_0_types.h $(KERNEL_DIR)/gemv_q8_0.cuh \
              $(KERNEL_DIR)/ops.cuh \
              $(GGUF_DIR)/gguf_parser.h $(TOK_DIR)/tokenizer.h

fence: $(BUILD_DIR)/fence
	@echo "Build complete: $(BUILD_DIR)/fence"
	@echo "Run: $(BUILD_DIR)/fence --model path/to/model.gguf"

$(BUILD_DIR)/fence: $(FENCE_DEPS) $(BUILD_DIR)/gguf_parser.o $(BUILD_DIR)/tokenizer.o | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRC_DIR)/main.cu $(MODEL_DIR)/qwen3.cu \
	    $(BUILD_DIR)/gguf_parser.o $(BUILD_DIR)/tokenizer.o

# ---- Tests ----
test_q6k_dot: $(BUILD_DIR)/test_q6k_dot
	@./$(BUILD_DIR)/test_q6k_dot

$(BUILD_DIR)/test_q6k_dot: $(TEST_DIR)/test_q6k_dot.cu \
                            $(KERNEL_DIR)/q6k_dot.cuh $(KERNEL_DIR)/q6k_types.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

test_gguf_parser: $(BUILD_DIR)/test_gguf_parser
ifndef MODEL
	$(error MODEL is not set. Use: make test_gguf_parser MODEL=path/to/model.gguf)
endif
	@./$(BUILD_DIR)/test_gguf_parser $(MODEL)

$(BUILD_DIR)/test_gguf_parser: $(TEST_DIR)/test_gguf_parser.cpp \
                                $(BUILD_DIR)/gguf_parser.o | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -o $@ $< $(BUILD_DIR)/gguf_parser.o

test_gemv_q6k: $(BUILD_DIR)/test_gemv_q6k
ifndef MODEL
	$(error MODEL is not set. Use: make test_gemv_q6k MODEL=path/to/model.gguf)
endif
	@./$(BUILD_DIR)/test_gemv_q6k $(MODEL)

$(BUILD_DIR)/test_gemv_q6k: $(TEST_DIR)/test_gemv_q6k.cu \
                             $(KERNEL_DIR)/gemv_q6k.cuh $(KERNEL_DIR)/q6k_dot.cuh \
                             $(KERNEL_DIR)/q6k_types.h $(BUILD_DIR)/gguf_parser.o | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(BUILD_DIR)/gguf_parser.o

# test_q6k_dot does not need a model file
test: test_q6k_dot

# ---- Debug forward-pass tool ----
debug_forward: $(BUILD_DIR)/debug_forward
	@echo "Run: $(BUILD_DIR)/debug_forward path/to/model.gguf"

$(BUILD_DIR)/debug_forward: $(TEST_DIR)/debug_forward.cu \
                             $(FENCE_DEPS) \
                             $(BUILD_DIR)/gguf_parser.o $(BUILD_DIR)/tokenizer.o | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(TEST_DIR)/debug_forward.cu $(MODEL_DIR)/qwen3.cu \
	    $(BUILD_DIR)/gguf_parser.o $(BUILD_DIR)/tokenizer.o

# ---- Clean ----
clean:
	rm -rf $(BUILD_DIR)
