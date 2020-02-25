NEW  := 4x4armv7
ASM  := GEMM4x4MicroV7.S
CC   := arm-linux-gnueabihf-gcc -mfpu=neon
AS   := arm-linux-gnueabihf-as -mfpu=neon
# NEW  := 4x4i8v1
# ASM  := GEMM4x4Micro.S
# CC   := aarch64-linux-gnu-gcc
# AS   := aarch64-linux-gnu-as

LINKER     := $(CC)
# CFLAGS     :=  -g -Wall
CFLAGS     := -Wall -O3 -flax-vector-conversions  
LDFLAGS    := -lm

UTIL       := copy_matrix.o \
              compare_matrices.o \
              random_matrix.o \
              dclock.o \
              print_matrix.o

TEST_OBJS  := test_MMult.o  

all: 
	make clean;
	make test_MMult.x

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h 
	$(AS) $(ASM) -o gemm.o
	$(CC) ${CFLAGS} $(TEST_OBJS) gemm.o $(NEW).c $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
