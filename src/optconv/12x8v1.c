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

#define CONV_H    (3*8)  // GEMM_R
#define CONV_W    (4*8)  // GEMM_R
#define CONV_K    (256)  // GEMM_P
#define CONV_C    (32)  // GEMM_Q
#define RS        (9)  // GEMM_Q
#define R         (3)  // GEMM_Q
#define W_UNROLL (4)
#define H_UNROLL (3)

void kernel12x8(int hout, int wout, int n, int k, float* sa, float * sb, float* sc, int hsout, int csout); 

float* fastMalloc(int size){
    void* ptr = 0;
    int iRet = posix_memalign(&ptr, 64, size * sizeof(float));
    assert(0 == iRet);
    return ptr;
}

void packB_k8(int cin, int cout, float* from, float* to) {

    float *src[8];
    float *dst = to;
    float *pos;
    float32x4_t val[9];

    for(int k=0; k<cout; k+=8){
        src[0] = from + (k + 0) * cin * RS;
        src[1] = from + (k + 1) * cin * RS;
        src[2] = from + (k + 2) * cin * RS;
        src[3] = from + (k + 3) * cin * RS;
        src[4] = from + (k + 4) * cin * RS;
        src[5] = from + (k + 5) * cin * RS;
        src[6] = from + (k + 6) * cin * RS;
        src[7] = from + (k + 7) * cin * RS;
        for(int c=0; c<cin;c++){
            int tc = c / CONV_C;
            int ic = c % CONV_C;
            dst = to + tc * CONV_C * cout * RS + k * CONV_C * RS + ic * RS * 8;
            for(int rs=0; rs<RS; rs++){
                *dst++ = *(src[0]++);
                *dst++ = *(src[1]++);
                *dst++ = *(src[2]++);
                *dst++ = *(src[3]++);
                *dst++ = *(src[4]++);
                *dst++ = *(src[5]++);
                *dst++ = *(src[6]++);
                *dst++ = *(src[7]++);
            
            }
        }
    }
}


void packA_12(int channel, int height, int width, int csin, int hsin, float* from, float* to) {

    float *src[3];
    float *dst = to;
    float *pos;
    float32x4_t val[9];

    // h * w = 12
    for(int h=0; h<height; h+=3){
        for(int w=0; w<width; w+=4){
            pos = from + w + h * hsin;
            src[0] = pos;
            src[1] = pos + hsin;
            src[2] = pos + hsin * 2;
            for(int c=0; c<channel; c++){
                for(int r=0; r<3; r++){
                    val[0] = vld1q_f32(src[0]);
                    val[1] = vld1q_f32(src[0] + 1);
                    val[2] = vld1q_f32(src[0] + 2);
                    src[0] += hsin;

                    val[3] = vld1q_f32(src[1]);
                    val[4] = vld1q_f32(src[1] + 1);
                    val[5] = vld1q_f32(src[1] + 2);
                    src[1] += hsin;

                    val[6] = vld1q_f32(src[2]);
                    val[7] = vld1q_f32(src[2] + 1);
                    val[8] = vld1q_f32(src[2] + 2);
                    src[2] += hsin;

                    vst1q_f32(dst,     val[0]);
                    vst1q_f32(dst + 4, val[3]);
                    vst1q_f32(dst + 8, val[6]);
                    vst1q_f32(dst + 12,val[1]);
                    vst1q_f32(dst + 16,val[4]);
                    vst1q_f32(dst + 20,val[7]);
                    vst1q_f32(dst + 24,val[2]);
                    vst1q_f32(dst + 28,val[5]);
                    vst1q_f32(dst + 32,val[8]);
                    dst += 36;
                }
                src[0] += csin - hsin * 3;
                src[1] += csin - hsin * 3;
                src[2] += csin - hsin * 3;
            }
        }
    }
}

float *sa=NULL, *sb=NULL;
void MY_IM_GEMM(int cin, int cout, int hout, int wout,
            float * restrict a,
            float * restrict b,
            float * restrict c
) {
#ifdef DEBUG_PRINT_DATA
    printf("\n-------\n");
    print_matrix(m, k, a, lda);
    printf("\n-------\n");
    print_matrix(cout, cin * 9, b, cin * 9);
    printf("\n-------\n");
#endif
 

    if(sa==NULL) 
        sa = fastMalloc(CONV_H * CONV_W * CONV_C * RS);
    //already padded
    const int pad = 1;
    int htile, wtile, ctile; 
    int hsin = wout + pad * 2;
    int csin = hsin * (hout + pad * 2);
    int csout = hout * wout;
    //assume hout % 3 == 0, wout % 4 == 0
    for (int hb = 0; hb < hout; hb += CONV_H) {
        htile = CONV_H <= hout-hb? CONV_H : hout-hb; 
        for (int wb = 0; wb < wout; wb += CONV_W) {
            wtile = CONV_W <= wout-wb? CONV_W : wout-wb;

            for (int cb = 0; cb < cin; cb += CONV_C){
                ctile = CONV_C <= cin-cb? CONV_C : cin-cb;
                float *w_ptr = b + cb * cout * RS;
                int ktile = CONV_K <= cout? CONV_K : cout;

                // micro kernel, split A Block to smaller Panel
                for (int hpos=0; hpos<htile; hpos += H_UNROLL) {
                    int microw;
                    for (int wpos=0; wpos<wtile;wpos += W_UNROLL) {
                        microw = wtile-wpos;
                        //if(microw >= 2 * W_UNROLL) {
                        //    microw = 2 * W_UNROLL;
                        //} else
						 if(microw > W_UNROLL) {
                            microw = W_UNROLL;
                        }

                        // coninueous packA
                        
                        packA_12(ctile, H_UNROLL, microw, csin, hsin, a + cb * csin + (hb+hpos)*hsin + wb+wpos, sa + (wpos*H_UNROLL + hpos * wtile)*ctile*RS);

                        kernel12x8(H_UNROLL, microw, ktile, ctile*RS, sa + (wpos*H_UNROLL + hpos * wtile)*ctile*RS, w_ptr, c + (hb+hpos)*wout + wb+wpos, wout, csout);
                    }
                }

                // the first B Block has been packed, proc the others 
                //for (ns = min_n; ns < n; ns += min_n) {
                //    min_n = n - ns;
                //    if (min_n >= GEMM_N * 2) {
                //        min_n = GEMM_N; 
                //    } else if(min_n > GEMM_N) {
                //        min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
                //    }

                //    kernel12x8(min_m, min_n, min_k, sa, sb, c + ms * ldc + ns, ldc);
                //}
            }
        }
    }
}

void kernel12x8(int hout, int wout, int cout, int crs, float* sa, float * sb, float* sc, int hsout, int csout) 
{
    assert(hout % 3 == 0 && wout % 4 == 0);

    float *restrict a = sa, *restrict b = sb, *restrict c = sc;
//#if __aarch64__
    int h_offset = hsout * sizeof(float);
    int c_offset = csout * sizeof(float) - h_offset * 2;
    int h,w,j;
    for(j = 0; j < cout; j += 8) {
        c = sc + j * csout;
        for(h = 0; h < hout; h += 3) {
            for(w = 0; w < wout; w += 4) {
            asm volatile (
                ".macro INIT12x8                     \n"
                "   mov x9,        %2               \n"
                "   ld1 {v8.4s},  [x9], %5         \n"
                "   ld1 {v16.4s},  [x9], %5         \n"
                "   ld1 {v24.4s}, [x9], %3         \n"
                "   ld1 {v9.4s}, [x9], %5         \n"
                "   ld1 {v17.4s}, [x9], %5         \n"
                "   ld1 {v25.4s}, [x9], %3         \n"
                "   ld1 {v10.4s}, [x9], %5         \n"
                "   ld1 {v18.4s}, [x9], %5         \n"
                "   ld1 {v26.4s}, [x9], %3         \n"
                "   ld1 {v11.4s}, [x9], %5         \n"
                "   ld1 {v19.4s}, [x9], %5         \n"
                "   ld1 {v27.4s}, [x9], %3         \n"
                "   ld1 {v12.4s},  [x9], %5         \n"
                "   ld1 {v20.4s},  [x9], %5         \n"
                "   ld1 {v28.4s}, [x9], %3         \n"
                "   ld1 {v13.4s}, [x9], %5         \n"
                "   ld1 {v21.4s}, [x9], %5         \n"
                "   ld1 {v29.4s}, [x9], %3         \n"
                "   ld1 {v14.4s}, [x9], %5         \n"
                "   ld1 {v22.4s}, [x9], %5         \n"
                "   ld1 {v30.4s}, [x9], %3         \n"
                "   ld1 {v15.4s}, [x9], %5         \n"
                "   ld1 {v23.4s}, [x9], %5         \n"
                "   ld1 {v31.4s}, [x9]             \n"
                ".endm                              \n" 
                "                                   \n"
                ".macro SAVE12x8                     \n"
                "   mov x9,        %2               \n"
                "   st1 {v8.4s},  [x9], %5         \n"
                "   st1 {v16.4s},  [x9], %5         \n"
                "   st1 {v24.4s}, [x9], %3         \n"
                "   st1 {v9.4s}, [x9], %5         \n"
                "   st1 {v17.4s}, [x9], %5         \n"
                "   st1 {v25.4s}, [x9], %3         \n"
                "   st1 {v10.4s}, [x9], %5         \n"
                "   st1 {v18.4s}, [x9], %5         \n"
                "   st1 {v26.4s}, [x9], %3         \n"
                "   st1 {v11.4s}, [x9], %5         \n"
                "   st1 {v19.4s}, [x9], %5         \n"
                "   st1 {v27.4s}, [x9], %3         \n"
                "   st1 {v12.4s},  [x9], %5         \n"
                "   st1 {v20.4s},  [x9], %5         \n"
                "   st1 {v28.4s}, [x9], %3         \n"
                "   st1 {v13.4s}, [x9], %5         \n"
                "   st1 {v21.4s}, [x9], %5         \n"
                "   st1 {v29.4s}, [x9], %3         \n"
                "   st1 {v14.4s}, [x9], %5         \n"
                "   st1 {v22.4s}, [x9], %5         \n"
                "   st1 {v30.4s}, [x9], %3         \n"
                "   st1 {v15.4s}, [x9], %5         \n"
                "   st1 {v23.4s}, [x9], %5         \n"
                "   st1 {v31.4s}, [x9]             \n"
                ".endm                              \n" 
                "                                   \n"
                //"   prfm pldl1keep, [%0]            \n"
                //"   prfm pldl1keep, [%1]            \n"
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   ld1 {v2.4s}, [%1], #16          \n"
                "INIT12x8                            \n"
                "mov x8,%4                          \n"
                "run:                               \n"
            
                "   fmla v8.4s , v2.4s, v0.s[0]     \n"
                "   ld1 {v1.4s}, [%0], #16          \n"
                "   fmla v9.4s , v2.4s, v0.s[1]     \n"
                "   fmla v10.4s, v2.4s, v0.s[2]     \n"
                "   ld1 {v3.4s}, [%1], #16          \n"
                "   fmla v11.4s, v2.4s, v0.s[3]     \n"
            
                "   fmla v12.4s, v2.4s, v1.s[0]     \n"
                "   ld1 {v4.4s}, [%1], #16          \n"
                "   fmla v13.4s, v2.4s, v1.s[1]     \n"
                "   fmla v14.4s, v2.4s, v1.s[2]     \n"
                "   fmla v15.4s, v2.4s, v1.s[3]     \n"
            
                "   fmla v16.4s, v3.4s, v0.s[0]     \n"
                "   ld1 {v2.4s}, [%1], #16          \n"
                "   fmla v17.4s, v3.4s, v0.s[1]     \n"
                "   fmla v18.4s, v3.4s, v0.s[2]     \n"
                "   fmla v19.4s, v3.4s, v0.s[3]     \n"
            

                "   fmla v20.4s, v3.4s, v1.s[0]     \n"
                "   fmla v21.4s, v3.4s, v1.s[1]     \n"
                "   fmla v22.4s, v3.4s, v1.s[2]     \n"
                "   fmla v23.4s, v3.4s, v1.s[3]     \n"
                "   subs x8, x8, #1                 \n"
            
                "   fmla v24.4s, v4.4s, v0.s[0]     \n"
                "   fmla v25.4s, v4.4s, v0.s[1]     \n"
                "   fmla v26.4s, v4.4s, v0.s[2]     \n"
                "   fmla v27.4s, v4.4s, v0.s[3]     \n"
            
                "   fmla v28.4s, v4.4s, v1.s[0]     \n"
                "   ld1 {v0.4s}, [%0], #16          \n"
                "   fmla v29.4s, v4.4s, v1.s[1]     \n"
                "   fmla v30.4s, v4.4s, v1.s[2]     \n"
                "   fmla v31.4s, v4.4s, v1.s[3]     \n"
                "   bne run                         \n"
                "SAVE12x8                            \n"
                "                                   \n"
                : "=r"(b),
                  "=r"(a),
                  "=r"(c),
                  "=r"(c_offset),
                  "=r"(crs),
                  "=r"(h_offset)
                : "0"(b),
                  "1"(a),
                  "2"(c),
                  "3"(c_offset),
                  "4"(crs),
                  "5"(h_offset)
                : "memory", "cc", "x8", "x9","x14", 
                "v0", "v1", "v2", "v3", "v4", 
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",  
                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", 
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );  

                c += 4;
                a -= 4;
                b = sb;
            } // endw
            c += hsout * 3 - wout;
        } //endh
        a = sa;
        sb += 8 * crs;
        b = sb;
    }// endj
}


