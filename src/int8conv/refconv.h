#ifndef CONV_REF_H
#define CONV_REF_H
#include <stdint.h>
#include <math.h>



static inline int8_t int32ToInt8(int data, int bias, float scale) {
    float value = (float)(data + bias) * scale;
    value       = value>-128.0f?value:-128.0f;
    value       = value<127.0f ?value:127.0f;
    return (int8_t)(roundf(value));
}


static void convi8_ref(const int8_t *src, int8_t *dst, const int8_t *weight, int32_t*  bias, float * scale, int height, int width, int cin, int cout){
	for(int h=0; h<height; h++){
		for(int w=0; w<width; w++){
			for(int o=0; o<cout; o++){
				int acc = 0;
				for(int i=0; i<cin; i++){
                    acc += weight[o*cin + i] * src[h*width*cin + w*cin + i];
				}
                dst[h*width*cout + w * cout + o] = int32ToInt8(acc, bias[o], scale[o]);
			}
		}
	}
}


#endif
 
