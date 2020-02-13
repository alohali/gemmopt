#ifndef __aarch64__

//#include "RPNAsmGlobal.h"

.text
.align 5


.macro asm_function fname
.global \fname
#ifdef __ELF__
.hidden \fname
.type \fname, %function
#endif
\fname:
.endm

.align 5
asm_function GEMM2x2Micro 
//void GEMM2x2Micro(int8_t* src, const int8_t* weight, int8_t* dst, int src_w_step, int dst_depth, 
//                            int cdiv8, float *scale, int32_t*bias)
//x0(src),
//x1(weight),
//x2(dst),
//x3(src_w_step),
//x4(dst_depth),
//x5(cdiv8),
//x6(scale),
//x7(bias)




sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64


//prefetch data
//assume buffer c>=16, even c==8
ld1 {v12.16b, v13.16b}, [x1], #32 
add x11, x0, x3 

cmp x5, #2
ld1 {v8.16b}, [x0], #16 
ld1 {v9.16b}, [x11], #16 

blt C8First 

C16Start:
    sub x5, x5, #2 
    smull v0.8h, v12.8b, v8.8b 
    smull v1.8h, v13.8b, v8.8b 
    smull v4.8h, v12.8b, v9.8b 
    smull v5.8h, v13.8b, v9.8b 
    smlal2 v0.8h, v12.16b, v8.16b 
    smlal2 v1.8h, v13.16b, v8.16b 
    smlal2 v4.8h, v12.16b, v9.16b 
    smlal2 v5.8h, v13.16b, v9.16b 
    saddlp v16.4s, v0.8h 
    saddlp v17.4s, v1.8h 
    saddlp v20.4s, v4.8h 
    saddlp v21.4s, v5.8h 
    cmp x5, #2
    ld1 {v8.16b}, [x0], #16 
    ld1 {v9.16b}, [x11], #16 
    ld1 {v12.16b, v13.16b}, [x1], #32 
     
    blt C8Last 
      
    C16Loop: 
        sub x5, x5, #2 
        smull v0.8h, v12.8b, v8.8b 
        smull v1.8h, v13.8b, v8.8b 
        smull v4.8h, v12.8b, v9.8b 
        smull v5.8h, v13.8b, v9.8b 
        smlal2 v0.8h, v12.16b, v8.16b 
        smlal2 v1.8h, v13.16b, v8.16b 
        smlal2 v4.8h, v12.16b, v9.16b 
        smlal2 v5.8h, v13.16b, v9.16b 
        sadalp v16.4s, v0.8h 
        sadalp v17.4s, v1.8h 
        sadalp v20.4s, v4.8h 
        sadalp v21.4s, v5.8h 
        cmp x5, #2 
        ld1 {v8.16b}, [x0], #16 
        ld1 {v9.16b}, [x11], #16 
        ld1 {v12.16b, v13.16b}, [x1], #32 
        bge C16Loop 
 
C8Last:
    cmp x5, #1
    blt LoopEnd 
    smull v0.8h, v12.8b, v8.8b 
    smull v1.8h, v13.8b, v8.8b 
    sadalp v16.4s, v0.8h 
    smull v4.8h, v12.8b, v9.8b 
    sadalp v17.4s, v1.8h 
    smull v5.8h, v13.8b, v9.8b 
    sadalp v20.4s, v4.8h 
    sadalp v21.4s, v5.8h 
    b LoopEnd
      
C8First:
    cmp x5, #1
    blt LoopEnd 
    smull v0.8h, v12.8b, v8.8b 
    smull v1.8h, v13.8b, v8.8b 
    saddlp v16.4s, v0.8h 
    smull v4.8h, v12.8b, v9.8b 
    saddlp v17.4s, v1.8h 
    smull v5.8h, v13.8b, v9.8b 
    saddlp v20.4s, v4.8h 
    saddlp v21.4s, v5.8h 
LoopEnd: 
      
     ld1 {v0.2s}, [x7], #8
     mov  v0.2d[1], v0.2d[0]
     addp v4.4s, v16.4s, v17.4s 
     addp v5.4s, v20.4s, v21.4s 
     addp v12.4s, v4.4s, v5.4s 
     add v16.4s, v12.4s, v0.4s 
     ld1 {v1.2s}, [x6], #8
     mov  v1.2d[1], v1.2d[0]
     scvtf v4.4s, v16.4s 
     fmul v12.4s, v4.4s, v1.4s 
      
     fcvtas v8.4s, v12.4s 
     sqxtn v0.4h, v8.4s 
     sqxtn v2.8b, v0.8h 
     umov w8, v2.s[0]
     strh w8, [x2]
     add x2, x2, x4
     lsr w8, w8, #16
     strh w8, [x2]  


sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
