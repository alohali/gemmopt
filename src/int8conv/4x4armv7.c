#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("arm neon not supported")
#endif

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>


// 4 channel pack
// a00~a03, a10~a13, a04~a07, a14~b17; a20~a23, a30~a33; a24~a27, a34~a37
void packB(int cin, int cout, int8_t *from, int8_t *to)
{

    int32_t *src = (int32_t *)from;
    int32_t *dst = (int32_t *)to;
    cin = cin / 4;
    for (int o = 0; o < cout; o += 4)
    {
        for (int c = 0; c < cin; c+=2)
        {
            dst[0] = src[(o + 0) * cin + c];
            dst[1] = src[(o + 1) * cin + c];
            dst[2] = src[(o + 0) * cin + c + 1];
            dst[3] = src[(o + 1) * cin + c + 1];
            dst[4] = src[(o + 2) * cin + c];
            dst[5] = src[(o + 3) * cin + c];
            dst[6] = src[(o + 2) * cin + c + 1];
            dst[7] = src[(o + 3) * cin + c + 1];
            dst += 8;
        }
    }
}

//assume c % 8 == 0, w % 4 == 0
// 4 line pack
// a00~a03, a10~a13, a04~a07, a14~b17; a20~a23, a30~a33; a24~a27, a34~a37
void pack_line_armv7_cpp(int cin, int8_t *from, int8_t *to)
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

void pack_line_armv7(int cin, const int32_t *src, int32_t *dst)
{
    cin = cin / 4;
    int temp[8];
    for (int c = 0; c < cin; c+=2)
    {
        int32x2x2_t v[2];
        v[0].val[0] = vld1_s32(src + c + 0 * cin);
        v[0].val[1] = vld1_s32(src + c + 1 * cin);
        v[1].val[0] = vld1_s32(src + c + 2 * cin);
        v[1].val[1] = vld1_s32(src + c + 3 * cin);
        vst2_s32(dst + c * 4, v[0]);
        vst2_s32(dst + c * 4 + 4, v[1]);
    }
}

int8_t *temp = NULL;
void kernel4x4(int cin, int cout, int hout, int wout, int8_t* sa, int8_t * sb, int8_t* sc, float *scale, int32_t *bias) 
{
    if(!temp){
        temp = (int8_t *)malloc(2048 * 4);
    }
    int8_t *a = sa, *b = sb, *c = sc;
    int cdiv8 = cin/8;
    //warning: different from frame work here !!!!!!!!!!!!!!!!!!!!!!
    int w_cstride = cdiv8 * 8;
    for(int h = 0; h < hout; h ++) {
        for(int w = 0; w < wout; w += 4)
        {
            pack_line_armv7(cin, a, temp);
            // pack_line_armv7_cpp(cin, a, temp);
            for (int j = 0; j < cout; j += 4)
            {
                GEMM4x4Micro(temp, b, c, cin, cout, cdiv8, scale + j, bias + j);
                c += 4;
                b += 4 * w_cstride;
            } // endo
            b = sb;
            c += cout * 3;
            a += cin * 4;
        } //endw
    }// endh
}


