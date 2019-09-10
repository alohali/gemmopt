#ifndef CONV_REF_H
#define CONV_REF_H


typedef struct 
{
  //do not change this
	/* data */
	int width;
	int height;
	int channel;
}dimM;


const int isCRSK = 0;
const int isCorr=1;
void convCPU(float *srca, dimM adim, float *srcb, dimM bdim, float *dst,dimM odim,const int uv, const int pad, const int batch){
	int a_cs = adim.height * adim.width;
	int a_hs = adim.width;
	int o_cs = odim.height * odim.width;
	int o_hs = odim.width;
	float a, b, acc;
	int b_ks, b_cs, b_hs, b_ws;
	if (isCRSK) {
		b_ks = 1;
		b_cs = bdim.width * bdim.height * odim.channel;
		b_hs = bdim.width * odim.channel;
		b_ws = odim.channel;
	}
	else {//KCRS
		b_ks = bdim.width * bdim.height * adim.channel;
		b_cs = bdim.width * bdim.height;
		b_hs = bdim.width;
		b_ws = 1;
	}
	for(int n=0;n<batch; ++n){                                              ///n
		for(int k=0;k<odim.channel; ++k){                     				///k
			for (int h = 0; h<odim.height; ++h){                            ///h
				for (int w = 0; w<odim.width; ++w){                         ///w
					acc = 0.0;
					for (int c = 0; c < adim.channel; c++) {                ///c
						for (int r = 0; r < bdim.height; r++) {             ///r
							for (int s = 0; s < bdim.width; s++) {          ///s
								if ((unsigned)(h + r - pad) < (unsigned)adim.height && (unsigned)(w + s - pad) < (unsigned)adim.width) {
									a = srca[c*a_cs + (h + r - pad)*a_hs + w + s - pad];
								}
								else {
									a = 0;
								}
								if (isCorr) {
									b = srcb[c*b_cs + r*b_hs + s * b_ws + k*b_ks];
								}
								else {
									b = srcb[c*b_cs + (bdim.height - 1 - r)*b_hs + (bdim.width - 1 - s) * b_ws + k*b_ks];
								}
								acc += a * b;
							}// for s
						}//for r
					}// for c
					dst[k * o_cs + h * o_hs + w] = acc;
				}//for w
			}// for h
		}// for k
		srca += a_cs * adim.channel;
		dst  += o_cs * odim.channel;
	}//for n
}


void ref_conv(int cout, int hout, int wout, int cin, float *src, float *w, float *dst)
{
	dimM a,b,c;
    a.width = wout + 2; a.height = hout + 2; a.channel = cin;
    b.width = 3; b.height = 3; b.channel = cin;
    c.width = wout; c.height = hout; c.channel = cout;
    convCPU(src, a, w, b, dst, c, 1, 0, 1);

}


#endif
 
