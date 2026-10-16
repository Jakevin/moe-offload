CXX = clang++
LLAMA_DIR ?= ../llama.cpp
BUILD_DIR = $(LLAMA_DIR)/build

CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter \
	-I$(LLAMA_DIR)/include \
	-I$(LLAMA_DIR)/ggml/include \
	-I$(LLAMA_DIR)/common \
	-I$(LLAMA_DIR)/vendor

LDFLAGS = -L$(BUILD_DIR)/bin \
	-Wl,-rpath,$(abspath $(BUILD_DIR)/bin) \
	-lllama-common -lllama -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas -lpthread

all: route_trace moe_loader

route_trace: route_trace.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

moe_loader: moe_loader.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f route_trace moe_loader
