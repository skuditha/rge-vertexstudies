CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -m64 -pthread -rdynamic
HIPOINC  := -I$(HIPO)/include/hipo4
HIPOLIB  := -L$(HIPO)/lib -lhipo4
ROOTINC  := $(shell root-config --cflags)
ROOTLIB  := $(shell root-config --libs)
SRC := src/hipo_to_root.cpp
BIN := build/rge_hipo_to_root
all: $(BIN)
$(BIN): $(SRC)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(HIPOINC) $(ROOTINC) $< -o $@ $(HIPOLIB) $(ROOTLIB)
clean:
	rm -rf build