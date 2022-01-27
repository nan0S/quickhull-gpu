TARGET 	  := ConvexHull

SRC_DIR   := src
BUILD_DIR := build

INC       := -I$(CUDA_HOME)/include -I./$(SRC_DIR)
LIB       := -L$(CUDA_HOME)/lib64 -lcudart -lcurand -lGLEW -lGL -lglfw -lGLU

VERSION   := -std=c++17
CXXFLAGS  := $(VERSION) -O2 -flto -fomit-frame-pointer -s -DNDEBUG
NVCCFLAGS := $(VERSION) -arch=sm_50 --ptxas-options=-O2 --use_fast_math -Wno-deprecated-gpu-targets

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
	nvcc -MMD -MP $(NVCCFLAGS) -c -o $@ $< $(INC)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
