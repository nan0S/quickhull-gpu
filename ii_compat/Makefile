TARGET 	  := ConvexHull

SRC_DIR   := src
BUILD_DIR := build
DEPS_DIR  := deps

INC       := -I./$(SRC_DIR) -I./$(DEPS_DIR) -I$(CUDA_HOME)/include
LIB       := -L$(CUDA_HOME)/lib64 -lcudart -lcurand -lGLEW -lGL -lglfw -lGLU

CXXFLAGS  := -std=c++17 -O2 -flto -fomit-frame-pointer -s -DNDEBUG
NVCCFLAGS := -std=c++14 -arch=sm_61 --ptxas-options=-O2 --use_fast_math -Wno-deprecated-gpu-targets

SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')
OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPENDS := $(patsubst %.cpp,$(BUILD_DIR)/%.d,$(SOURCES))
SOURCES := $(shell find $(SRC_DIR) -name '*.cu')
OBJECTS += $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SOURCES))
DEPENDS += $(patsubst %.cu,$(BUILD_DIR)/%.d,$(SOURCES))

.PHONY: clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	g++ $(CXXFLAGS) -o $@ $^ $(INC) $(LIB)

-include $(DEPENDS)

$(BUILD_DIR)/%.o: %.cpp Makefile
	@mkdir -p $(shell dirname $@)
	g++ -MMD -MP $(CXXFLAGS) -c -o $@ $< $(INC)

$(BUILD_DIR)/%.o: %.cu Makefile
	@mkdir -p $(shell dirname $@)
	nvcc $(NVCCFLAGS) -c -o $@ $< $(INC)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
