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


//prefetch data
//assume buffer c>=16, even c==8
vld1.8 {q10, q11}, [r1]!
add r8, r0, r3 

cmp r5, #2
vld1.8 {q8}, [r0]!
vld1.8 {q9}, [r8]!

blt C8First 

C16Start:
    sub r5, r5, #2 
    vmull.s8 q0, d20, d16 
    vmull.s8 q1, d22, d16 
    vmull.s8 q2, d20, d18
    vmull.s8 q3, d22, d18
    vmlal.s8 q0, d21, d17 
    vmlal.s8 q1, d23, d17 
    vmlal.s8 q2, d21, d19 
    vmlal.s8 q3, d23, d19 
    vpaddl.s16 q4, q0 
    vpaddl.s16 q5, q1 
    vpaddl.s16 q6, q2 
    vpaddl.s16 q7, q3 
    cmp r5, #2
    vld1.8 {q8}, [r0]!
    vld1.8 {q9}, [r8]!
    vld1.8 {q10, q11}, [r1]!
     
    blt C8Last 
      
    C16Loop: 
        sub r5, r5, #2 
        vmull.s8 q0, d20, d16 
        vmull.s8 q1, d22, d16 
        vmull.s8 q2, d20, d18
        vmull.s8 q3, d22, d18
        vmlal.s8 q0, d21, d17 
        cmp r5, #2 
        vmlal.s8 q1, d23, d17 
        vmlal.s8 q2, d21, d19 
        vld1.8 {q8}, [r0]! 
        vmlal.s8 q3, d23, d19 
        vpadal.s16 q4, q0 
        vld1.8 {q9}, [r8]! 
        vpadal.s16 q5, q1 
        vld1.8 {q10, q11}, [r1]!
        vpadal.s16 q6, q2 
        vpadal.s16 q7, q3 
        bge C16Loop 
 
C8Last:
    cmp r5, #1
    blt LoopEnd 
    vmull.s8 q0, d20, d16 
    vmull.s8 q1, d22, d16 
    vpadal.s16 q4, q0 
    vmull.s8 q2, d20, d18
    vpadal.s16 q5, q1 
    vmull.s8 q3, d22, d18
    vpadal.s16 q6, q2 
    vpadal.s16 q7, q3 
    b LoopEnd
      
C8First:
    cmp r5, #1
    blt LoopEnd 
    vmull.s8 q0, d20, d16 
    vmull.s8 q1, d22, d16 
    vpaddl.s16 q4, q0 
    vmull.s8 q2, d20, d18
    vpaddl.s16 q5, q1 
    vmull.s8 q3, d22, d18
    vpaddl.s16 q6, q2 
    vpaddl.s16 q7, q3 
LoopEnd: 
      
    vld1.8 d4, [r7]!
    vmov.s32 q0, 0x3f000000
    vpadd.s32 d8, d10 
    vpadd.s32 d9, d11 
    vpadd.s32 d12, d14 
    vpadd.s32 d13, d15 
    vadd.s32 d8, d9 
    vadd.s32 d12, d13 
    
    vadd.s32 d8, d8, d4 
    vadd.s32 d9, d12, d4
    vld1.8 d6, [r6]!
    vcvt.f32.s32 q4, q4 
    vmla.f32 d0, d8, d6
    vmla.f32 d1, d9, d6

    vcvtr.s32.f32 q0, q0

    vqmovn.s32 d2,q0 
    vqmovn.s16 d4,q1 
    
    vmov.32 r8, d4[0]
    strh r8, [r2]
    add r2, r2, r4
    lsr r8, r8, #16
    strh r8, [r2]  


vpop {q4-q7}
pop {r4, r5, r6, r7, r8, pc}


#endif
