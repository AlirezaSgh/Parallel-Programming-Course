# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CC := g++-14
else
    CC := g++
endif

# Compiler and flags
CFLAGS := -c -fopenmp
CPPFLAGS := -I/opt/homebrew/opt/libomp/include -I../Include
LNKFLAGS := 
LIBDIR := -L/opt/homebrew/opt/libomp/lib
LIBFILE := -lomp

# Targets
all: clean main clean_obj run

main: main.o 
	$(CC) $(LIBDIR) $(LIBFILE) main.o -o main.out

main.o: main.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) main.cpp

clean_obj:
	rm -f *.o

clean: 
	rm -f main.out

run:
	./main.out
