NEW  := 4x4i8v1
#NEW  := MMult_4x4_17

#
# sample makefile
#

CC         := gcc
LINKER     := $(CC)
#CFLAGS     :=  -g -Wall
CFLAGS     := -Wall -O2 -flax-vector-conversions  
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
	as GEMM4x4Micro.s -o gemm.o
	$(LINKER) $(TEST_OBJS) gemm.o $(NEW).c $(UTIL) $(LDFLAGS) \
        $(BLAS_LIB) -o $(TEST_BIN) $@ 

run:	
	make all
	export OMP_NUM_THREADS=1
	export GOTO_NUM_THREADS=1
	echo "version = '$(NEW)';" > output_$(NEW).m
	./test_MMult.x >> output_$(NEW).m

clean:
	rm -f *.o *~ core *.x

cleanall:
	rm -f *.o *~ core *.x output*.m *.eps *.png
