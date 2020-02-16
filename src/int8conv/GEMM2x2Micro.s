#ifdef __aarch64__

#include "RPNAsmGlobal.h"

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
//r0(src),
//r1(weight),
//r2(dst),
//r3(src_w_step),
//r4(dst_depth),
//r5(cdiv8),
//r6(scale),
//r7(bias)

push {r4, r5, r6, r7, r8, lr}

ldr r4, [sp, #24]
ldr r5, [sp, #28]
ldr r6, [sp, #32]
ldr r7, [sp, #36]

vpush {q4-q7}

add r8, r0, r3 
cmp r5, #2

//prefetch data
//assume buffer c>=16, even c==8
vld1.8 {q10, q11}, [r1]!
vld1.8 {q12, q13}, [r1]!

vld1.8 {q14}, [r0]!
vld1.8 {q15}, [r8]!

blt C8First 

C16Start:
    sub r5, r5, #2 
    vmull.s8 q0, d20, d28 
    vmull.s8 q1, d22, d28 
    vmlal.s8 q0, d21, d29 
    vmlal.s8 q1, d23, d29 
    vpaddl.s16 q2, q0 
    vpaddl.s16 q3, q1 

    vmull.s8 q0, d20, d30
    vmull.s8 q1, d22, d30
    vmlal.s8 q0, d21, d31 
    vmlal.s8 q1, d23, d31 
    vpaddl.s16 q4, q0 
    vpaddl.s16 q5, q1 

    vmull.s8 q0, d24, d28 
    vmull.s8 q1, d26, d28 
    vmlal.s8 q0, d25, d29 
    vmlal.s8 q1, d27, d29 
    vpaddl.s16 q6, q0 
    vpaddl.s16 q7, q1 

    vmull.s8 q0, d24, d30
    vmull.s8 q1, d26, d30
    vmlal.s8 q0, d25, d31 
    vmlal.s8 q1, d27, d31 
    vpaddl.s16 q8, q0 
    vpaddl.s16 q9, q1 

    cmp r5, #2
    vld1.8 {q14}, [r0]!
    vld1.8 {q15}, [r8]!
    vld1.8 {q10, q11}, [r1]!
    vld1.8 {q12, q13}, [r1]!
     
    blt C8Last 
      
    C16Loop: 
        sub r5, r5, #2 
        vmull.s8 q0, d20, d28 
        vmull.s8 q1, d22, d28 
        vmlal.s8 q0, d21, d29 
        vmlal.s8 q1, d23, d29 
        vpadal.s16 q2, q0 
        vpadal.s16 q3, q1 

        vmull.s8 q0, d20, d30
        vmull.s8 q1, d22, d30
        vmlal.s8 q0, d21, d31 
        vmlal.s8 q1, d23, d31 
        vpadal.s16 q4, q0 
        vpadal.s16 q5, q1 

        vmull.s8 q0, d24, d28 
        vmull.s8 q1, d26, d28 
        vmlal.s8 q0, d25, d29 
        vmlal.s8 q1, d27, d29 
        vpadal.s16 q6, q0 
        vpadal.s16 q7, q1 

        vmull.s8 q0, d24, d30
        vmull.s8 q1, d26, d30
        vmlal.s8 q0, d25, d31 
        vmlal.s8 q1, d27, d31 
        vpadal.s16 q8, q0 
        vpadal.s16 q9, q1 


        cmp r5, #2 


        vld1.8 {q14}, [r0]! 
        vld1.8 {q15}, [r8]! 
        vld1.8 {q10, q11}, [r1]!
        vld1.8 {q12, q13}, [r1]!
        bge C16Loop 
 
C8Last:
    cmp r5, #1
    blt LoopEnd 
    vmull.s8 q0, d20, d28 
    vmull.s8 q1, d22, d28 
    vpadal.s16 q2, q0 
    vpadal.s16 q3, q1 
    vmull.s8 q0, d20, d30
    vmull.s8 q1, d22, d30
    vpadal.s16 q4, q0 
    vpadal.s16 q5, q1 
    vmull.s8 q0, d24, d28 
    vmull.s8 q1, d26, d28 
    vpadal.s16 q6, q0 
    vpadal.s16 q7, q1 
    vmull.s8 q0, d24, d30
    vmull.s8 q1, d26, d30
    vpadal.s16 q8, q0 
    vpadal.s16 q9, q1 
    b LoopEnd
      
C8First:
    cmp r5, #1
    blt LoopEnd 
    vmull.s8 q0, d20, d28 
    vmull.s8 q1, d22, d28 
    vpaddl.s16 q2, q0 
    vpaddl.s16 q3, q1 

    vmull.s8 q0, d20, d30
    vmull.s8 q1, d22, d30
    vpaddl.s16 q4, q0 
    vpaddl.s16 q5, q1 

    vmull.s8 q0, d24, d28 
    vmull.s8 q1, d26, d28 
    vpaddl.s16 q6, q0 
    vpaddl.s16 q7, q1 

    vmull.s8 q0, d24, d30
    vmull.s8 q1, d26, d30
    vpaddl.s16 q8, q0 
    vpaddl.s16 q9, q1 
LoopEnd: 
    //bias q14, scale q15
    vld1.8 {q14}, [r7]
    vmov.s32 q0, 0x3f000000
    vld1.8 {q15}, [r6]
    vmov.s32 q1, 0x3f000000
    //q2 ~ q9  --> q2, q3
    vpadd.s32 d4, d5
    vpadd.s32 d6, d7 
    vpadd.s32 d8, d9 
    vpadd.s32 d10, d11

    vpadd.s32 d12, d13 
    vpadd.s32 d14, d15 
    vpadd.s32 d16, d17 
    vpadd.s32 d18, d19

    vpadd.s32 d4, d4, d6 
    vpadd.s32 d5, d8, d10 
    vpadd.s32 d6, d12, d14 
    vpadd.s32 d7, d16, d18 
    
    vqadd.s32 q2, q14 
    vqadd.s32 q3, q14 

    //(q2, q3 + bias) * scale --> q0, q1
    vcvt.f32.s32 q2, q2 
    vcvt.f32.s32 q3, q3 

    vmla.f32 q0, q2, q15
    vmla.f32 q1, q3, q15

    vcvt.s32.f32 q0, q0
    vcvt.s32.f32 q1, q1

    //q0, q1 --> q4 --> d10
    vqmovn.s32 d8,q0
    vqmovn.s32 d9,q1
    vqmovn.s16 d10,q4 
    
    //vmov.32 r8, d4[0]
    vst1.s32 d10[0], [r2]
    add r2, r2, r4
    vst1.s32 d10[1], [r2]

vpop {q4-q7}
pop {r4, r5, r6, r7, r8, pc}


#endif
