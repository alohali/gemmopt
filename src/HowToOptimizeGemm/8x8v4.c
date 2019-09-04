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

/* Create macros so that the matrices are stored in row-major order */

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

#define min(i, j) ((i) < (j) ? (i): (j))

/**
About GEMM_K or kc:
1. mc = kc, since we have to maxmize (2 * mc * kc/(2 * mc + kc))
2. The equation exists provided kc << n.
3. mc * kc <= K

About GEMM_M or mc:
1. The larger mc * nc, the better calculation efficiency
2. We prepare to load A into L2 cache. Avoiding TLB miss (which would
stall CPU), subset of A should remains so until no longer needed.

About KENEL_4x4, mr=4 and nr=4
In order to move data efficiently to the registers.
Here we use C_block = A_panel x Transpose(B_panel)

In accordance to page.14 "6. MOE DETAILS YET",




L1d cahce = 32K, and L2 cache = 2MB. `getconf -a | grep PAGESIZE` = 4096.
Thus L1d is not the Cannikin, it is constraint to page size.

min_nn * kc <= PAGESIZE/2,  4 <= min_nn <= 12, so that 170 <= kc <= 512, we use 256.
After reading 6.4, rk3399 L2 cache is large, mc = 1MB / 256 = 4096 


*/
#define GEMM_N (256)  // GEMM_R
#define GEMM_M (512)  // GEMM_P
#define GEMM_K (256)  // GEMM_Q
#define GEMM_UNROLL (8)
#define KERNEL_8x8  kernel_8x8_v1

/* Routine for computing C = A * B + C */
void packB_8(int k, int n, float* from, int ldb, float* to);
void packA_8(int m, int k, float* from, int lda, float* to);
void packA_4(int m, int k, float* from, int lda, float* to);
void kernel_8x8_v1(int m, int n, int k, 
        float* sa, float* sb, float* sc, int ldc);

float* fastMalloc(int size){
    void* ptr = 0;
    int iRet = posix_memalign(&ptr, 64, size * sizeof(float));
    assert(0 == iRet);
    return ptr;
}
void packA_4(int m, int k, float* from, int lda, float* to) {
#ifdef DEBUG_PACK_SHAPE
    printf("\n packA_4, m=%d, k=%d", m, k);
#endif
    assert( k != 0 && m != 0 && k % 4 == 0 && m % 4 == 0);
    int i, j;

    float *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    float *a_offset8, *a_offset5, *a_offset6, *a_offset7;
    float *b_offset;
    float  ctemp1,  ctemp2,  ctemp3,  ctemp4;
    float  ctemp5,  ctemp6,  ctemp7,  ctemp8;
    float  ctemp9, ctemp10, ctemp11, ctemp12;
    float ctemp13, ctemp14, ctemp15, ctemp16;

    a_offset = from;
    b_offset = to;

    j = (m >> 3);
    do{
        a_offset1  = a_offset;
        a_offset2  = a_offset1 + lda;
        a_offset3  = a_offset2 + lda;
        a_offset4  = a_offset3 + lda;
        a_offset5  = a_offset4 + lda;
        a_offset6  = a_offset5 + lda;
        a_offset7  = a_offset6 + lda;
        a_offset8  = a_offset7 + lda;
        a_offset += 8 * lda;

        i = (k >> 1);
        do{
            ctemp1  = *(a_offset1 + 0);
            ctemp2  = *(a_offset1 + 1);
            ctemp3  = *(a_offset2 + 0);
            ctemp4  = *(a_offset2 + 1);

            ctemp5  = *(a_offset3 + 0);
            ctemp6  = *(a_offset3 + 1);
            ctemp7  = *(a_offset4 + 0);
            ctemp8  = *(a_offset4 + 1);

            ctemp9  = *(a_offset5 + 0);
            ctemp10 = *(a_offset5 + 1);
            ctemp11 = *(a_offset6 + 0);
            ctemp12 = *(a_offset6 + 1);

            ctemp13 = *(a_offset7 + 0);
            ctemp14 = *(a_offset7 + 1);
            ctemp15 = *(a_offset8 + 0);
            ctemp16 = *(a_offset8 + 1);

            *(b_offset +  0) = ctemp1;
            *(b_offset +  1) = ctemp3;
            *(b_offset +  2) = ctemp5;
            *(b_offset +  3) = ctemp7;

            *(b_offset +  4) = ctemp9;
            *(b_offset +  5) = ctemp11;
            *(b_offset +  6) = ctemp13;
            *(b_offset +  7) = ctemp15;

            *(b_offset +  8) = ctemp2;
            *(b_offset +  9) = ctemp4;
            *(b_offset + 10) = ctemp6;
            *(b_offset + 11) = ctemp8;

            *(b_offset + 12) = ctemp10;
            *(b_offset + 13) = ctemp12;
            *(b_offset + 14) = ctemp14;
            *(b_offset + 15) = ctemp16;

            a_offset1 += 2;
            a_offset2 += 2;
            a_offset3 += 2;
            a_offset4 += 2;
            a_offset5 += 2;
            a_offset6 += 2;
            a_offset7 += 2;
            a_offset8 += 2;

            b_offset += 16;
            i --;
        }while(i > 0);
        j --;
    }while(j > 0);
}
float *sa=NULL, *sb=NULL;
/* Suppose that m%4==0 and n%4==0 and k%4==0, avoiding process boundary !! */
void MY_MMult(int m, int n, int k, float * restrict a, int lda,
                                   float * restrict b, int ldb,
                                   float * restrict c, int ldc )
{
#ifdef DEBUG_PRINT_DATA
    printf("\n-------\n");
    print_matrix(m, k, a, lda);
    printf("\n-------\n");
    print_matrix(k, n, b, ldb);
    printf("\n-------\n");
#endif
 

    if(sa==NULL) {
        sa = fastMalloc(800*800);
        sb = fastMalloc(800*800);
    }
    int ms, mms, ns, ks;
    int min_m, min_mm, min_n, min_k;
    for (ms = 0; ms < m; ms += GEMM_M) {
        min_m = m - ms;
        if (min_m > GEMM_M) {
            min_m = GEMM_M;
        }

        for (ks = 0; ks < k; ks += min_k){
            min_k = k - ks;
            if (min_k >= (GEMM_K << 1)) {
                min_k = GEMM_K;
            } else if (min_k > GEMM_K) {
                min_k = (min_k / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            }

            // first packB
            min_n = n;
            if (n >= GEMM_N * 2) {
                min_n = GEMM_N;
            } else if(n > GEMM_N) {
                min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            }
            packB_8(min_k, min_n, b + ks * ldb, ldb, sb);

            // micro kernel, split A Block to smaller Panel
            for (mms = ms; mms < ms + min_m; mms += min_mm) {
                min_mm = (ms + min_m) - mms;
                if(min_mm >= 2 * GEMM_UNROLL) {
                    min_mm = 2 * GEMM_UNROLL;
                } else if(min_mm > GEMM_UNROLL) {
                    min_mm = GEMM_UNROLL;
                }

                // coninueous packA
                packA_4(min_mm, min_k, a + mms * lda + ks, lda, sa + min_k * (mms - ms));

                KERNEL_8x8(min_mm, min_n, min_k, sa + min_k * (mms - ms), sb, c + mms * ldc, ldc);
            }

            // the first B Block has been packed, proc the others 
            for (ns = min_n; ns < n; ns += min_n) {
                min_n = n - ns;
                if (min_n >= GEMM_N * 2) {
                    min_n = GEMM_N; 
                } else if(min_n > GEMM_N) {
                    min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
                }

                packB_8(min_k, min_n, b + ns + ldb * ks, ldb, sb);
                KERNEL_8x8(min_m, min_n, min_k, sa, sb, c + ms * ldc + ns, ldc);
            }
        }
    }
}

/**

float* a: A
float* b: (B)T
float* c: C

C = A * (B)T

A1 A2 A3    B1 B4 B7
A4 A5 A6  x B2 B5 B8 => C1 C4 C7 C2 C5 C8 C3 C6 C9 (packed)
A7 A8 A9    B3 B4 B9

Calculation sequence:
1st. calculate C1
2st. calculate C4
3st. calculate C7
...
9st. calculate C9

A1-A9/B1-B9 is packed block, not single number.
C1-C9 is 4x4 block, not single number.

Output
C1 C2 C3
C4 C5 C6
C7 C8 C9

 */
void kernel_8x8_v1(int m, int n, int k,
    float* sa, float * sb, float* sc, int ldc) {
    assert(m > 0 && n > 0 && k > 0);
    assert(m % 8 == 0 && n % 8 == 0 && k % 8 == 0);

    float *restrict a = sa, *restrict b = sb, *restrict c = sc;
//#if __aarch64__
#define ASMTEST 1 
#if ASMTEST
    int ldc_offset = ldc * sizeof(float);
#endif
    int i, j, l;
    for(i = 0; i < m; i += 8) {
        for(j = 0; j < n; j += 8) {
#if ASMTEST 
asm volatile (
    ".macro INIT8x8                     \n"
    "   mov x9,        %2               \n"
    "   add x13,       x9 , #16         \n"
    "   ld1 {v16.4s}, [x9], %3          \n"
    "   ld1 {v17.4s}, [x9], %3          \n"
    "   ld1 {v18.4s}, [x9], %3          \n"
    "   ld1 {v19.4s}, [x9], %3          \n"
    "   ld1 {v20.4s}, [x9], %3          \n"
    "   ld1 {v21.4s}, [x9], %3          \n"
    "   ld1 {v22.4s}, [x9], %3          \n"
    "   ld1 {v23.4s}, [x9]              \n"
    "   ld1 {v24.4s}, [x13], %3         \n"
    "   ld1 {v25.4s}, [x13], %3         \n"
    "   ld1 {v26.4s}, [x13], %3         \n"
    "   ld1 {v27.4s}, [x13], %3         \n"
    "   ld1 {v28.4s}, [x13], %3         \n"
    "   ld1 {v29.4s}, [x13], %3         \n"
    "   ld1 {v30.4s}, [x13], %3         \n"
    "   ld1 {v31.4s}, [x13]             \n"
    ".endm                              \n" 
    "                                   \n"
    ".macro SAVE8x8                     \n"
    "   mov x9,        %2               \n"
    "   add x13,       x9 , #16         \n"
    "   st1 {v16.4s}, [x9], %3          \n"
    "   st1 {v17.4s}, [x9], %3          \n"
    "   st1 {v18.4s}, [x9], %3          \n"
    "   st1 {v19.4s}, [x9], %3          \n"
    "   st1 {v20.4s}, [x9], %3          \n"
    "   st1 {v21.4s}, [x9], %3          \n"
    "   st1 {v22.4s}, [x9], %3          \n"
    "   st1 {v23.4s}, [x9]              \n"
    "   st1 {v24.4s}, [x13], %3         \n"
    "   st1 {v25.4s}, [x13], %3         \n"
    "   st1 {v26.4s}, [x13], %3         \n"
    "   st1 {v27.4s}, [x13], %3         \n"
    "   st1 {v28.4s}, [x13], %3         \n"
    "   st1 {v29.4s}, [x13], %3         \n"
    "   st1 {v30.4s}, [x13], %3         \n"
    "   st1 {v31.4s}, [x13]             \n"
    ".endm                              \n" 
    "                                   \n"
    //"   prfm pldl1keep, [%0]            \n"
    //"   prfm pldl1keep, [%1]            \n"
    "INIT8x8                            \n"
    "mov x8,%4                          \n"
    "run:                               \n"
    "   ld1 {v0.4s}, [%0], #16          \n"
    "   ld1 {v2.4s}, [%1], #16          \n"

    "   fmla v16.4s, v0.4s, v2.s[0]     \n"
    "   ld1 {v3.4s}, [%1], #16          \n"
    "   fmla v17.4s, v0.4s, v2.s[1]     \n"
    "   fmla v18.4s, v0.4s, v2.s[2]     \n"
    "   ld1 {v1.4s}, [%0], #16          \n"
    "   fmla v19.4s, v0.4s, v2.s[3]     \n"

    "   fmla v20.4s, v0.4s, v3.s[0]     \n"
    "   prfm pldl1keep, [%0, #64]       \n"
    "   fmla v21.4s, v0.4s, v3.s[1]     \n"
    "   fmla v22.4s, v0.4s, v3.s[2]     \n"
    "   prfm pldl1keep, [%1, #64]       \n"
    "   fmla v23.4s, v0.4s, v3.s[3]     \n"

    "   fmla v24.4s, v1.4s, v2.s[0]     \n"
    "   subs x8, x8, #1                 \n"
    "   fmla v25.4s, v1.4s, v2.s[1]     \n"
    "   prfm pldl1keep, [%0, #128]       \n"
    "   fmla v26.4s, v1.4s, v2.s[2]     \n"
    "   fmla v27.4s, v1.4s, v2.s[3]     \n"
    "   prfm pldl1keep, [%1, #128]       \n"

    "   fmla v28.4s, v1.4s, v3.s[0]     \n"
    "   prfm pldl1keep, [%0, #192]       \n"
    "   fmla v29.4s, v1.4s, v3.s[1]     \n"
    "   fmla v30.4s, v1.4s, v3.s[2]     \n"
    "   prfm pldl1keep, [%1, #192]       \n"
    "   fmla v31.4s, v1.4s, v3.s[3]     \n"
    "   bne run                         \n"
    "SAVE8x8                            \n"
    "                                   \n"
    : "=r"(b),
      "=r"(a),
      "=r"(c),
      "=r"(ldc_offset),
      "=r"(k)
    : "0"(b),
      "1"(a),
      "2"(c),
      "3"(ldc_offset),
      "4"(k)
    : "memory", "cc", "x8", "x9","x13", "x14", 
    "v0", "v1", "v2", "v3",  
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", 
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
);  
#else
            __builtin_prefetch(b, 0, 3);
            __builtin_prefetch(a, 0, 3);

            float32x4_t v16 = (vld1q_f32(c));
            float32x4_t v17 = (vld1q_f32(c + ldc));
            float32x4_t v18 = (vld1q_f32(c + 2*ldc));
            float32x4_t v19 = (vld1q_f32(c + 3*ldc));
            float32x4_t v20 = (vld1q_f32(c + 4*ldc));
            float32x4_t v21 = (vld1q_f32(c + 5*ldc));
            float32x4_t v22 = (vld1q_f32(c + 6*ldc));
            float32x4_t v23 = (vld1q_f32(c + 7*ldc));
            float32x4_t v24 = (vld1q_f32(c + 4));
            float32x4_t v25 = (vld1q_f32(c + 4 + ldc));
            float32x4_t v26 = (vld1q_f32(c + 4 + 2*ldc));
            float32x4_t v27 = (vld1q_f32(c + 4 + 3*ldc));
            float32x4_t v28 = (vld1q_f32(c + 4 + 4*ldc));
            float32x4_t v29 = (vld1q_f32(c + 4 + 5*ldc));
            float32x4_t v30 = (vld1q_f32(c + 4 + 6*ldc));
            float32x4_t v31 = (vld1q_f32(c + 4 + 7*ldc));

            for(l = 0; l < k; l++) {
                float32x4_t v0 = vld1q_f32(b);
                float32x4_t v1 = vld1q_f32(b+4);
                float32x4_t v2 = vld1q_f32(a);
                float32x4_t v3 = vld1q_f32(a+4);

                __builtin_prefetch(b+16, 0, 3);
                __builtin_prefetch(a+16, 0, 3);

                v16 = vmlaq_laneq_f32(v16, v0, v2, 0);
                v17 = vmlaq_laneq_f32(v17, v0, v2, 1);
                v18 = vmlaq_laneq_f32(v18, v0, v2, 2);
                v19 = vmlaq_laneq_f32(v19, v0, v2, 3);

                v20 = vmlaq_laneq_f32(v20, v0, v3, 0);
                v21 = vmlaq_laneq_f32(v21, v0, v3, 1);
                v22 = vmlaq_laneq_f32(v22, v0, v3, 2);
                v23 = vmlaq_laneq_f32(v23, v0, v3, 3);

                v24 = vmlaq_laneq_f32(v24, v1, v2, 0);
                v25 = vmlaq_laneq_f32(v25, v1, v2, 1);
                v26 = vmlaq_laneq_f32(v26, v1, v2, 2);
                v27 = vmlaq_laneq_f32(v27, v1, v2, 3);

                v28 = vmlaq_laneq_f32(v28, v1, v3, 0);
                v29 = vmlaq_laneq_f32(v29, v1, v3, 1);
                v30 = vmlaq_laneq_f32(v30, v1, v3, 2);
                v31 = vmlaq_laneq_f32(v31, v1, v3, 3);


                b += 8;
                a += 8;
            } // endl
            

            vst1q_f32(c, v16);
            vst1q_f32(c + ldc, v17);
            vst1q_f32(c + 2 * ldc, v18);
            vst1q_f32(c + 3 * ldc, v19);
            vst1q_f32(c + 4 * ldc, v20);
            vst1q_f32(c + 5 * ldc, v21);
            vst1q_f32(c + 6 * ldc, v22);
            vst1q_f32(c + 7 * ldc, v23);

            vst1q_f32(c + 4, v24);
            vst1q_f32(c + 4 + ldc, v25);
            vst1q_f32(c + 4 + 2 * ldc, v26);
            vst1q_f32(c + 4 + 3 * ldc, v27);
            vst1q_f32(c + 4 + 4 * ldc, v28);
            vst1q_f32(c + 4 + 5 * ldc, v29);
            vst1q_f32(c + 4 + 6 * ldc, v30);
            vst1q_f32(c + 4 + 7 * ldc, v31);

#endif

            c += 8;
            a -= 8*k;
        } // endj
        sc += ldc*8;
        c = sc;
        a += 8*k;
        b = sb;
    }// endi
}


/**
pack A means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag

Output:
0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7
8 8 8 8 9 9 9 9 a a a a b b b b 
c c c c d d d d e e e e f f f f

Draw it with a line
*/
void packA_8(int m, int k, float* from, int lda, float* to) {
    float *src[8];
    float *dst = to;
    float32x2_t val[8];
    float32x2_t tem[8];

    for(int i=0; i<m; i+=8){
        src[0] = from + i * lda;
        src[1] = src[0] + lda;
        src[2] = src[1] + lda;
        src[3] = src[2] + lda;
        src[4] = src[3] + lda;
        src[5] = src[4] + lda;
        src[6] = src[5] + lda;
        src[7] = src[6] + lda;
        for(int j=0; j<k/2; j++){
           val[0] = vld1_f32(src[0]); 
           src[0] += 2;
           val[1] = vld1_f32(src[1]); 
           src[1] += 2;
           val[2] = vld1_f32(src[2]); 
           src[2] += 2;
           val[3] = vld1_f32(src[3]); 
           src[3] += 2;
           val[4] = vld1_f32(src[4]); 
           src[4] += 2;
           val[5] = vld1_f32(src[5]); 
           src[5] += 2;
           val[6] = vld1_f32(src[6]); 
           src[6] += 2;
           val[7] = vld1_f32(src[7]); 
           src[7] += 2;
           tem[0] = vtrn1_f32(val[0],val[1]);
           tem[1] = vtrn2_f32(val[0],val[1]);
           tem[2] = vtrn1_f32(val[2],val[3]);
           tem[3] = vtrn2_f32(val[2],val[3]);
           tem[4] = vtrn1_f32(val[4],val[5]);
           tem[5] = vtrn2_f32(val[4],val[5]);
           tem[6] = vtrn1_f32(val[6],val[7]);
           tem[7] = vtrn2_f32(val[6],val[7]);
           vst1_f32(dst, tem[0]); 
           vst1_f32(dst + 2, tem[2]); 
           vst1_f32(dst + 4, tem[4]); 
           vst1_f32(dst + 6, tem[6]); 
           vst1_f32(dst + 8, tem[1]); 
           vst1_f32(dst + 10, tem[3]); 
           vst1_f32(dst + 12, tem[5]); 
           vst1_f32(dst + 14, tem[7]); 
           dst += 16;
        }
    }
}

/*
suppose that k and n is mutiple of 4
pack B means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag, not like pack A

Output:
0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
8 9 a b 8 9 a b 8 9 a b 8 9 a b
4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7
c d e f c d e f c d e f c d e f
*/
void packB_8(int k, int n, float* from, int ldb, float* to) {

    assert( k != 0 && n != 0 && k % 4 == 0 && n % 4 == 0);
    float *src[4];
    float *dst;
    float32x4_t val[8];

    for(int j=0; j<k; j+=4){
        dst  = to + j * 8; 
        src[0] = from + j * ldb;
        src[1] = src[0] + ldb;
        src[2] = src[1] + ldb;
        src[3] = src[2] + ldb;
        for(int i=0; i<n/8; i++){
            val[0] = vld1q_f32(src[0]);
            val[1] = vld1q_f32(src[0] + 4);
            src[0] += 8;

            val[2] = vld1q_f32(src[1]);
            val[3] = vld1q_f32(src[1] + 4);
            src[1] += 8;

            val[4] = vld1q_f32(src[2]);
            val[5] = vld1q_f32(src[2] + 4);
            src[2] += 8;

            val[6] = vld1q_f32(src[3]);
            val[7] = vld1q_f32(src[3] + 4);
            src[3] += 8;

            vst1q_f32(dst,     val[0]);
            vst1q_f32(dst + 4, val[1]);
            vst1q_f32(dst + 8, val[2]);
            vst1q_f32(dst + 12,val[3]);
            vst1q_f32(dst + 16,val[4]);
            vst1q_f32(dst + 20,val[5]);
            vst1q_f32(dst + 24,val[6]);
            vst1q_f32(dst + 28,val[7]);
            dst += k * 8;
            
        }
    }


}
