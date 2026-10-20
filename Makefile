NVCC       ?= nvcc
NVCCFLAGS  ?= -O3 -std=c++17
ARCH       ?= native
SRC        := src/main.cu
HEADERS    := src/model.cuh src/rng.cuh
BIN        := raytrace
ARGS       ?=

all: raytrace

$(BIN): $(SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(SRC) -o $(BIN)

run: $(BIN)
	./$(BIN) $(ARGS)

clean:
	rm -f $(BIN)
