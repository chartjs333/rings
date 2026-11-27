CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic -ffp-contract=off
.PHONY: all test ubsan
all: librings.so libmulticlass.so mc_tests
librings.so: c_api.cpp rings.hpp c_api.h
	$(CXX) $(CXXFLAGS) -fPIC -shared c_api.cpp -o $@
libmulticlass.so: mc_api.cpp multiclass.hpp rings.hpp
	$(CXX) $(CXXFLAGS) -fPIC -shared mc_api.cpp -o $@
mc_tests: mc_tests.cpp multiclass.hpp rings.hpp
	$(CXX) $(CXXFLAGS) mc_tests.cpp -o $@
test: all
	./mc_tests
	python3 -m unittest test_multiclass.py
ubsan: mc_tests.cpp multiclass.hpp rings.hpp
	$(CXX) -std=c++17 -O1 -g -fsanitize=undefined -fno-sanitize-recover=undefined -ffp-contract=off mc_tests.cpp -o mc_tests_ubsan
	./mc_tests_ubsan
