#ifndef CONV_REF_H
#define CONV_REF_H
#include <stdint.h>
#include <math.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("arm neon not supported")
#endif


static inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = (float)(data + bias) * scale;
    value       = value>-128.0f?value:-128.0f;
    value       = value<127.0f ?value:127.0f;
    return (int8_t)(vcvtns_s32_f32(value));
}

//nhwc data, oirs weight
static void convi8_ref(const int8_t *src, int8_t *dst, const int8_t *weight, int32_t*  bias, 
                        float * scale, int height, int width, int cin, int cout, int hin, 
                        int win, int kernel, int pad, int stride){
    int8_t a, b;
	for(int h=0; h<height; h++){
		for(int w=0; w<width; w++){
			for(int o=0; o<cout; o++){
				int acc = 0;
				for(int i=0; i<cin; i++){
                    for(int r=0; r<kernel; r++){
                        for(int s=0; s<kernel; s++){
                            int srch = h * stride + r - pad;
                            int srcw = w * stride + s - pad;
                            if(srch>=0 && srcw>=0 && srch<hin && srcw<win){
                                a = src[srch*win*cin + srcw*cin+i];
                                b = weight[o*cin*kernel*kernel + i*kernel*kernel + r*kernel+s];
                                acc += a * b;
                            }
                        }
                    }
				}
                dst[h*width*cout + w * cout + o] = int32ToInt8(acc, bias[o], scale[o]);
			}
		}
	}
}


#endif
 
