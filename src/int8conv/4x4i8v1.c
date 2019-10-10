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


void packB_k8(int cin, int cout, int8_t* from, int8_t* to) {

    int64_t *src = (int64_t *)from;
    int64_t *dst = (int64_t *)to;
    cin = cin/8;
    for(int o=0; o<cout;o+=4){
        for(int i=0; i<cin; i++){
            dst[0] = src[o*cin];
            dst[1] = src[(o+1)*cin];
            dst[2] = src[(o+2)*cin];
            dst[3] = src[(o+3)*cin];
            dst += 4;
            src++;
        }
    }
    
}



void kernel4x4(int cin, int cout, int hout, int wout, int8_t* sa, int8_t * sb, int8_t* sc, float *scale, int32_t *bias) 
{
    int8_t *restrict a = sa, *restrict b = sb, *restrict c = sc;
    int cdiv16 = cin/16;
    for(int h = 0; h < hout; h ++) {
        for(int w = 0; w < wout; w += 4) {
            for(int j = 0; j < cout; j += 4) {
            	asm volatile (

                "mov x8, %0\n"
                "mov x14, %4\n"
                "mov x10, %1\n"
                "add x11, x8, %3\n"
                "add x12, x11, %3\n"
                "add x13, x12, %3\n"
                "ld1 {v12.16b, v13.16b}, [x10], #32\n"
                "ld1 {v14.16b, v15.16b}, [x10], #32\n"
                "ld1 {v8.16b}, [x8], #16\n"
                "ld1 {v9.16b}, [x11], #16\n"
                "ld1 {v10.16b}, [x12], #16\n"
                "ld1 {v11.16b}, [x13], #16\n"
                "subs x9, %5, #1\n"
                
                "smull v0.8h, v12.8b, v8.8b\n"
                "smull v1.8h, v13.8b, v8.8b\n"
                "smlal2 v0.8h, v12.16b, v8.16b\n"
                "smlal2 v1.8h, v13.16b, v8.16b\n"
                "saddlp v16.4s, v0.8h\n"
                "saddlp v17.4s, v1.8h\n"
                 
                "smull v2.8h, v14.8b, v8.8b\n"
                "smull v3.8h, v15.8b, v8.8b\n"
                "smull v4.8h, v12.8b, v9.8b\n"
                "smull v5.8h, v13.8b, v9.8b\n"
                "smull v6.8h, v14.8b, v9.8b\n"
                "smull v7.8h, v15.8b, v9.8b\n"
                "smlal2 v2.8h, v14.16b, v8.16b\n"
                "smlal2 v3.8h, v15.16b, v8.16b\n"
                "smlal2 v4.8h, v12.16b, v9.16b\n"
                "smlal2 v5.8h, v13.16b, v9.16b\n"
                "smlal2 v6.8h, v14.16b, v9.16b\n"
                "smlal2 v7.8h, v15.16b, v9.16b\n"
                "saddlp v18.4s, v2.8h\n"
                "saddlp v19.4s, v3.8h\n"
                "saddlp v20.4s, v4.8h\n"
                "saddlp v21.4s, v5.8h\n"
                "saddlp v22.4s, v6.8h\n"
                "saddlp v23.4s, v7.8h\n"
                 
                
                "smull v0.8h, v12.8b, v10.8b\n"
                "smull v1.8h, v13.8b, v10.8b\n"
                "smull v2.8h, v14.8b, v10.8b\n"
                "smull v3.8h, v15.8b, v10.8b\n"
                "smlal2 v0.8h, v12.16b, v10.16b\n"
                "smlal2 v1.8h, v13.16b, v10.16b\n"
                "smlal2 v2.8h, v14.16b, v10.16b\n"
                "smlal2 v3.8h, v15.16b, v10.16b\n"
                "saddlp v24.4s, v0.8h\n"
                "saddlp v25.4s, v1.8h\n"
                "saddlp v26.4s, v2.8h\n"
                "saddlp v27.4s, v3.8h\n"
                 
                "smull v4.8h, v12.8b, v11.8b\n"
                "smull v5.8h, v13.8b, v11.8b\n"
                "smull v6.8h, v14.8b, v11.8b\n"
                "smull v7.8h, v15.8b, v11.8b\n"
                "smlal2 v4.8h, v12.16b, v11.16b\n"
                "smlal2 v5.8h, v13.16b, v11.16b\n"
                "smlal2 v6.8h, v14.16b, v11.16b\n"
                "smlal2 v7.8h, v15.16b, v11.16b\n"
                "saddlp v28.4s, v4.8h\n"
                "saddlp v29.4s, v5.8h\n"
                "saddlp v30.4s, v6.8h\n"
                "saddlp v31.4s, v7.8h\n"
                 
                "beq L4LoopSzEnd\n"
                 
                "L4LoopSz:\n"
                "    ld1 {v12.16b, v13.16b}, [x10], #32\n"
                "    ld1 {v14.16b, v15.16b}, [x10], #32\n"
                "    ld1 {v8.16b}, [x8], #16\n"
                "    ld1 {v9.16b}, [x11], #16\n"
                "    ld1 {v10.16b}, [x12], #16\n"
                "    ld1 {v11.16b}, [x13], #16\n"
                "    smull v0.8h, v12.8b, v8.8b\n"
                "    smull v1.8h, v13.8b, v8.8b\n"
                "    smlal2 v0.8h, v12.16b, v8.16b\n"
                "    smlal2 v1.8h, v13.16b, v8.16b\n"
                "    sadalp v16.4s, v0.8h\n"
                "    sadalp v17.4s, v1.8h\n"
                 
                "    smull v2.8h, v14.8b, v8.8b\n"
                "    smull v3.8h, v15.8b, v8.8b\n"
                "    smull v4.8h, v12.8b, v9.8b\n"
                "    smull v5.8h, v13.8b, v9.8b\n"
                "    smull v6.8h, v14.8b, v9.8b\n"
                "    smull v7.8h, v15.8b, v9.8b\n"
                "    smlal2 v2.8h, v14.16b, v8.16b\n"
                "    smlal2 v3.8h, v15.16b, v8.16b\n"
                "    smlal2 v4.8h, v12.16b, v9.16b\n"
                "    smlal2 v5.8h, v13.16b, v9.16b\n"
                "    smlal2 v6.8h, v14.16b, v9.16b\n"
                "    smlal2 v7.8h, v15.16b, v9.16b\n"
                "    sadalp v18.4s, v2.8h\n"
                "    sadalp v19.4s, v3.8h\n"
                "    sadalp v20.4s, v4.8h\n"
                "    sadalp v21.4s, v5.8h\n"
                "    sadalp v22.4s, v6.8h\n"
                "    sadalp v23.4s, v7.8h\n"
                 
                "    smull v0.8h, v12.8b, v10.8b\n"
                "    smull v1.8h, v13.8b, v10.8b\n"
                "    smull v2.8h, v14.8b, v10.8b\n"
                "    smull v3.8h, v15.8b, v10.8b\n"
                "    smlal2 v0.8h, v12.16b, v10.16b\n"
                "    smlal2 v1.8h, v13.16b, v10.16b\n"
                "    smlal2 v2.8h, v14.16b, v10.16b\n"
                "    smlal2 v3.8h, v15.16b, v10.16b\n"
                "    sadalp v24.4s, v0.8h\n"
                "    sadalp v25.4s, v1.8h\n"
                "    sadalp v26.4s, v2.8h\n"
                "    sadalp v27.4s, v3.8h\n"
                 
                "    smull v4.8h, v12.8b, v11.8b\n"
                "    smull v5.8h, v13.8b, v11.8b\n"
                "    smull v6.8h, v14.8b, v11.8b\n"
                "    smull v7.8h, v15.8b, v11.8b\n"
                "    smlal2 v4.8h, v12.16b, v11.16b\n"
                "    smlal2 v5.8h, v13.16b, v11.16b\n"
                "    smlal2 v6.8h, v14.16b, v11.16b\n"
                "    smlal2 v7.8h, v15.16b, v11.16b\n"
                "    subs x9, x9, #1\n"
                "    sadalp v28.4s, v4.8h\n"
                "    sadalp v29.4s, v5.8h\n"
                "    sadalp v30.4s, v6.8h\n"
                "    sadalp v31.4s, v7.8h\n"
                "    bne L4LoopSz\n"
                 
                "L4LoopSzEnd:\n"
                 
                "ld1 {v0.4s}, [%7], #16\n"
                "addp v4.4s, v16.4s, v17.4s\n"
                "addp v5.4s, v18.4s, v19.4s\n"
                "addp v6.4s, v20.4s, v21.4s\n"
                "addp v7.4s, v22.4s, v23.4s\n"
                "addp v8.4s, v24.4s, v25.4s\n"
                "addp v9.4s, v26.4s, v27.4s\n"
                "addp v10.4s, v28.4s, v29.4s\n"
                "addp v11.4s, v30.4s, v31.4s\n"
                 
                "addp v12.4s, v4.4s, v5.4s\n"
                "addp v13.4s, v6.4s, v7.4s\n"
                "addp v14.4s, v8.4s, v9.4s\n"
                "addp v15.4s, v10.4s, v11.4s\n"
                "ld1 {v1.4s}, [%6], #16\n"
                "add v16.4s, v12.4s, v0.4s\n"
                "add v17.4s, v13.4s, v0.4s\n"
                "add v18.4s, v14.4s, v0.4s\n"
                "add v19.4s, v15.4s, v0.4s\n"
                 
                "scvtf v4.4s, v16.4s\n"
                "scvtf v5.4s, v17.4s\n"
                "scvtf v6.4s, v18.4s\n"
                "scvtf v7.4s, v19.4s\n"
                 
                "fmul v12.4s, v4.4s, v1.4s\n"
                "fmul v13.4s, v5.4s, v1.4s\n"
                "fmul v14.4s, v6.4s, v1.4s\n"
                "fmul v15.4s, v7.4s, v1.4s\n"
                 
                "fcvtzs v8.4s, v12.4s\n"
                "fcvtzs v9.4s, v13.4s\n"
                "fcvtzs v10.4s, v14.4s\n"
                "fcvtzs v11.4s, v15.4s\n"
                 
                "sqxtn v0.4h, v8.4s\n"
                "sqxtn2 v0.8h, v9.4s\n"
                "sqxtn v1.4h, v10.4s\n"
                "sqxtn2 v1.8h, v11.4s\n"
                "sqxtn v2.8b, v0.8h\n"
                "sqxtn2 v2.16b, v1.8h\n"
                "st1 {v2.s}[0], [x14], %4\n"
                "st1 {v2.s}[1], [x14], %4\n"
                "st1 {v2.s}[2], [x14], %4\n"
                "st1 {v2.s}[3], [x14]\n"                
                : "=r"(a),
                  "=r"(b),
                  "=r"(c),
                  "=r"(cin),
                  "=r"(cout),
                  "=r"(cdiv16),
                  "=r"(scale),
                  "=r"(bias)
                : "0"(a),
                  "1"(b),
                  "2"(c),
                  "3"(cin),
                  "4"(cout),
                  "5"(cdiv16),
                  "6"(scale),
                  "7"(bias)
                : "memory", "cc", "x8", "x9","x11","x12","x13","x14", 
                "v0", "v1", "v2", "v3", "v4","v5","v6","v7", 
                "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",  
                "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", 
                "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            	);  

                c += 4;
                b += 4 * cin;
            } // endo
            b = sb;
            c += wout * 3;
            a += cin * 4;
        } //endw
    }// endh
}


