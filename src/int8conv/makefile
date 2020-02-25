NEW  := 4x4armv7
ASM  := GEMM4x4MicroV7.S
CC   := arm-linux-gnueabihf-gcc
AS   := arm-linux-gnueabihf-as

LINKER     := $(CC)
CFLAGS     :=  -g -Wall
#CFLAGS     := -Wall -O3 -flax-vector-conversions  
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
	$(AS) $(ASM) -mfpu=neon -o gemm.o
	$(CC) -mfpu=neon ${CFLAGS} $(TEST_OBJS) gemm.o $(NEW).c $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
