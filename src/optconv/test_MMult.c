#include <stdio.h>
// #include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include "refconv.h"

#include "parameters.h"

void REF_MMult(int, int, int, float *, int, float *, int, float *, int );
void MY_IM_GEMM(int, int, int,int, float *, float *, float *);
void copy_matrix(int, int, float *, int, float *, int );
void random_matrix(int, int, float *, int, int);
float compare_matrices( int, int, float *, int, float *, int );
void packB_k8(int cin, int cout, float* from, float* to);

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

  float 
    *a, *b, *c, *cref, *bpack;    
  
  printf( "\nSize, Gflops\n" );
    
  int compare = 1;
  if(argc>1){
    compare = atoi(argv[1]);
  }
//  for ( p=PFIRST; p<=PLAST; p+=PINC ){
  {
    int cin = 64;
    int hout = 3*16;
    int hin  = hout + 2;
    int wout = 4*16;
    int win  = wout + 2; 
    int cout = 64;
    gflops = 2.0 * cin * cout * hout * wout * 9 * 1.0e-09;

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    a = ( float * ) malloc( hin * win * (cin+1) * sizeof( float ) );  
    b = ( float * ) malloc( cin * cout * 9 * sizeof( float ) );
    bpack = ( float * ) malloc( cin * cout * 10 * sizeof( float ) );
    c = ( float * ) malloc( hout * wout * cout * 2 * sizeof( float ) );
    cref = ( float * ) malloc( hout * wout * cout*2 * sizeof( float ) );

    /* Generate random matrices A, B, Cold */
    random_matrix( cin, hin*win, a, hin*win , 1);
    random_matrix( cout, cin * 9, b, cin * 9 , 1);
    packB_k8(cin, cout, b, bpack);
    ref_conv( cout, hout, wout, cin, a, b, cref);

    /* check output */
    for ( rep=0; rep<2; rep++ ){
	  memset(c, 0, hout*wout*cout*sizeof(float));
      MY_IM_GEMM( cin, cout, hout, wout, a,  bpack, c);
    }
    if(compare)
        diff = compare_matrices( cout, hout*wout, c, hout*wout, cref, hout*wout );
    if(diff > 0.05f || diff < -0.05f){
        compare = 0;
        diff = 0;
    }

    MY_IM_GEMM( cin, cout, hout, wout, a,  bpack, c);
    dtime = dclock();
    for ( rep=0; rep<NREPEATS; rep++ ){
      MY_IM_GEMM( cin, cout, hout, wout, a,  bpack, c);
    }

    dtime = dclock() - dtime;
    dtime /= NREPEATS;

    printf( "%d, %.3f\n", p, gflops / dtime);
    fflush( stdout );

    free( a );
    free( b );
    free( c );
    free( cref );
  }

  printf( "\n" );

  exit( 0 );
}

