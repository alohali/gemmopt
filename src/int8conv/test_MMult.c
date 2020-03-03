#include <stdio.h>
// #include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include "refconv.h"

#include "parameters.h"

void MY_IM_GEMM(int, int, int,int, int8_t *, int8_t *, int8_t *);
void copy_matrix(int, int, int8_t *, int, int8_t *, int );
void random_matrix(int, int, int8_t *, int, int);
int8_t compare_matrices( int, int, int8_t *, int, int8_t *, int );
void packB(int cin, int cout, int8_t* from, int8_t* to);

double dclock();

int main(int argc, char**argv)
{
  int 
    p, 
    rep;

  double 
    dtime, dtime_best,        
    gflops, 
    diff;

  int8_t 
    *a, *b, *c, *cref, *bpack;    
  int32_t *bias;
  float *scale;
  printf( "\nci co h w  t gop\n" );
    
  int compare = 1;
  if(argc>1){
    compare = atoi(argv[1]);
  }

  int testcase[][4] = {
      //cin cout h w
      // {256, 512, 3*32, 4*32},
      // {8, 4, 4, 4},
      {8, 8, 4, 4},
      {16, 8, 4, 4},
      {8, 16, 4, 4},
      // {8, 32, 4, 4},
      {16, 4, 4, 4},
      {8, 16, 96, 80},
      {16, 16, 96, 80},
      {72, 8, 96, 80},
      {24, 4, 96, 80},
      // {16, 16, 96, 80},

      // {128, 128, 56, 56},
      // {24, 32, 32, 32},
      // {8, 32, 32, 32},
      // {32, 32, 32, 32},
      // {32, 32, 16, 16},
      // {32, 32, 32, 32},
      // {32, 32, 64, 64},
      // {64, 64, 64, 64},
      // {64, 64, 16, 16},
      // {128, 128, 32, 32},
      // {128, 256, 28, 28},
      // {256, 256, 28, 28},
      // {512, 512, 12, 16},
      // {1024, 512, 3*16, 4*12},
      // {2048, 2048, 3*16, 4*12},
      // {64, 128, 3*16, 4*16},
      // /{512, 64, 3*32, 4*32},
      //mobilenet
      // {32, 64, 112, 112},
      // {64, 128, 56, 56},
      // {128, 128, 56, 56},
      // {128, 256, 28, 28},
      // {256, 256, 28, 28},
      // {256, 512, 49, 4},
      // {512, 512, 49, 4},
      // {512, 1024, 25, 2},
      // {1024, 1024, 25, 2},
  };
  for (int tid = 0; tid < sizeof(testcase) / sizeof(testcase[0]); tid++){
        int cin = testcase[tid][0];
  int cout = testcase[tid][1];
  int hout = testcase[tid][2];
  int hin = hout;
  int wout = testcase[tid][3];
  int win = wout;
  gflops = 2.0 * cin * cout * hout * wout * 1.0e-09;

  int c16 = (cin + 15) & (~15);
  a = (int8_t *)malloc(hin * win * (cin) * sizeof(int8_t) + 64);
  b = (int8_t *)malloc(c16 * cout * sizeof(int8_t) + 64);
  bpack = (int8_t *)malloc(c16 * cout * NREPEATS * sizeof(int8_t) + 64);
  memset(bpack, 0, c16 * cout * NREPEATS * sizeof(int8_t) + 64);
  c = (int8_t *)malloc(hout * wout * cout * sizeof(int8_t));
  cref = (int8_t *)malloc(hout * wout * cout * sizeof(int8_t));
  scale = (float *)malloc(cout * sizeof(float));
  bias = (int32_t *)malloc(cout * sizeof(int32_t));

  int random = 1;
  for (int si = 0; si < cout; si++)
  {
    if (random)
    {
      scale[si] = (float)((rand()) % 64) / 255.0;
      bias[si] = (rand() % 16);
    }
    else
    {
      scale[si] = 0.5;
      bias[si] = 0;
    }
    }

    printf("run %d %d %d %d: ", cin, cout, hout, wout);
    /* Generate random matrices A, B, Cold */
    random_matrix(hin * win, cin, a, cin, random);
    random_matrix(cout, cin, b, cin, random);
    convi8_ref(a, cref, b, bias, scale,   hout, wout, cin, cout);

    packB(cin, cout, b, bpack);  
    /* check output */
    for ( rep=0; rep<2; rep++ ){
	    memset(c, 0, hout*wout*cout*sizeof(int8_t));
      kernel4x4( cin, cout, hout, wout, a,  bpack, c, scale, bias);
    }
    if (compare)
      diff = compare_matrices(hout * wout, cout, c, cout, cref, cout);
    if (diff > 0.05f || diff < -0.05f)
    {
      compare = 0;
      diff = 0;
    }

    kernel4x4( cin, cout, hout, wout, a,  b, c, scale, bias);
    dtime = dclock();
    for ( rep=0; rep<NREPEATS; rep++ ){
      kernel4x4( cin, cout, hout, wout, a,  bpack + rep * cin * cout, c, scale, bias);
    }

    dtime = dclock() - dtime;
    dtime /= NREPEATS;

    printf( "%.3f %.3f\n", dtime*1000, gflops / dtime);
    fflush( stdout );

    free( a );
    free( bpack );
    free( b );
    free( c );
    free( cref );
    free( scale );
    free( bias );
  }

  printf( "\n" );

  exit( 0 );
}

