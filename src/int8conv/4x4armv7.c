#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("arm neon not supported")
#endif

#include <assert.h>
#include <stdlib.h>

/* Block sizes */
#define DEBUG_PACK_SHAPE
#undef DEBUG_PACK_SHAPE
#define DEBUG_PRINT_DATA
#undef DEBUG_PRINT_DATA


#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define min(i, j) ((i) < (j) ? (i): (j))

#define CONV_H    (3*4)  // GEMM_R
#define CONV_W    (4*4)  // GEMM_R
#define CONV_K    (256)  // GEMM_P
#define CONV_C    (32)  // GEMM_Q
#define RS        (9)  // GEMM_Q
#define R         (3)  // GEMM_Q
#define W_UNROLL (4)
#define H_UNROLL (3)


void packB(int cin, int cout, int8_t *from, int8_t *to)
{

    int32_t *src = (int32_t *)from;
    int32_t *dst = (int32_t *)to;
    cin = cin / 4;
    for (int o = 0; o < cout; o += 4)
    {
        for (int i = 0; i < cin; i++)
        {
            dst[0] = src[(o + 0) * cin + i];
            dst[1] = src[(o + 1) * cin + i];
            dst[2] = src[(o + 2) * cin + i];
            dst[3] = src[(o + 3) * cin + i];
            dst += 4;
        }
    }
}

//assume c % 8 == 0, w % 4 == 0
// 4 line pack
// a00~a03, a10~a13, a04~a07, a14~b17; a20~a23, a30~a33; a24~a27, a34~a37
void pack_line_armv7(int cin, int8_t *from, int8_t *to)
{
    int32_t *src = (int32_t *)from;
    int32_t *dst = (int32_t *)to;
    cin = cin / 4;
    for (int c = 0; c < cin; c+=2)
    {
        dst[0] = src[c + 0 * cin];
        dst[1] = src[c + 1 * cin];
        dst[2] = src[c + 0 * cin + 1];
        dst[3] = src[c + 1 * cin + 1];
        dst[4] = src[c + 2 * cin];
        dst[5] = src[c + 3 * cin];
        dst[6] = src[c + 2 * cin + 1];
        dst[7] = src[c + 3 * cin + 1];
        dst += 8;
    }
}

extern void GEMM2x2Micro(int8_t *a, const int8_t *b, int8_t *c, int cin, int cout, int cdiv8, float *scale, int32_t *bias);
int8_t *temp = NULL;
void kernel4x4(int cin, int cout, int hout, int wout, int8_t* sa, int8_t * sb, int8_t* sc, float *scale, int32_t *bias) 
{
    if(!temp){
        temp = (int8_t *)malloc(2048 * 4);
    }
    int8_t *a = sa, *b = sb, *c = sc;
    int cdiv8 = cin/8;
    int w_cstride = (cdiv8+1)/2*16;
    for(int h = 0; h < hout; h ++) {
        for(int w = 0; w < wout; w += 4) {
            pack_line_armv7(cin, a, temp);
            for(int j = 0; j < cout; j += 4) {
                GEMM2x2Micro(temp, b, c, cin, cout, cdiv8, scale + j, bias + j);
                c += 4;
                b += 4 * w_cstride;
            } // endo
            b = sb;
            c += cout * 3;
            a += cin * 4;
        } //endw
    }// endh
}


