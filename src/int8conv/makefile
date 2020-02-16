NEW  := 2x2i8v1
#NEW  := 4x4i8v1

#
# sample makefile
#

CC         := arm-linux-gnueabihf-gcc
LINKER     := $(CC)
#CFLAGS     :=  -g -Wall
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

# test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h 
# 	aarch64-linux-gnu-as GEMM2x2Microarmv8.S -o gemm22.o
# 	aarch64-linux-gnu-as GEMM4x4Micro.S -o gemm44.o
# 	$(LINKER) $(TEST_OBJS) gemm22.o gemm44.o $(NEW).c $(UTIL) $(LDFLAGS) \
#         $(BLAS_LIB) -o $(TEST_BIN) $@ 

test_MMult.x: $(TEST_OBJS) $(UTIL) parameters.h 
	arm-linux-gnueabihf-as GEMM2x2Micro.S -mfpu=neon -o gemm22.o
	$(LINKER) -mfpu=neon ${CFLAGS} $(TEST_OBJS) gemm22.o $(NEW).c $(UTIL) $(LDFLAGS) \
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
