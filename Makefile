CXX=g++
CXXFLAGS_BASE=-std=c++1z -march=broadwell -mtune=broadwell -Wall -Wextra
CXXFLAGS_RELEASE=-DNDEBUG -O3 -g3
CXXFLAGS_DEBUG=-DDEBUG -O0 -g3

DEBUG ?= 0
ifeq ($(DEBUG), 1)
    CXXFLAGS=$(CXXFLAGS_BASE) $(CXXFLAGS_DEBUG)
else
    CXXFLAGS=$(CXXFLAGS_BASE) $(CXXFLAGS_RELEASE)
endif

rt: rt.cpp
	$(CXX) -o $@ $< $(CPPFLAGS) $(CXXFLAGS)


.PHONY: run
run: rt
	time -p ./rt && geeqie foo.ppm

.PHONY: debug
gdb: CXXFLAGS=$(CXXFLAGS_BASE) $(CXXFLAGS_DEBUG)
gdb: clean rt
	gdb ./rt

.PHONY: clean
clean:
	rm -f rt
