#include <stdio.h>
// #include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include "parameters.h"

void REF_MMult(int, int, int, float *, int, float *, int, float *, int );
void MY_MMult(int, int, int, float *, int, float *, int, float *, int );
void copy_matrix(int, int, float *, int, float *, int );
void random_matrix(int, int, float *, int, int);
float compare_matrices( int, int, float *, int, float *, int );

double dclock();

int main()
{
  int 
    p, 
    m, n, k,
    lda, ldb, ldc, 
    rep;

  double 
    dtime, dtime_best,        
    gflops, 
    diff;

  float 
    *a, *b, *c, *cref, *cold;    
  
  printf( "\nSize, Gflops\n" );
    
  for ( p=PFIRST; p<=PLAST; p+=PINC ){
    m = ( M == -1 ? p : M );
    n = ( N == -1 ? p : N );
    k = ( K == -1 ? p : K );

    gflops = 2.0 * m * n * k * 1.0e-09;

    lda = ( LDA == -1 ? m : LDA );
    ldb = ( LDB == -1 ? k : LDB );
    ldc = ( LDC == -1 ? m : LDC );

    /* Allocate space for the matrices */
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    a = ( float * ) malloc( lda * (k+1) * sizeof( float ) );  
    b = ( float * ) malloc( ldb * n * sizeof( float ) );
    c = ( float * ) malloc( ldc * n * sizeof( float ) );
    cold = ( float * ) malloc( ldc * n * sizeof( float ) );
    cref = ( float * ) malloc( ldc * n * sizeof( float ) );

    /* Generate random matrices A, B, Cold */
    random_matrix( m, k, a, lda , 1);
    random_matrix( k, n, b, ldb , 1);
    random_matrix( m, n, cold, ldc, 1);
#if 1 
    memset(cold, 0, ldc * n * sizeof(float));
#endif

    copy_matrix( m, n, cold, ldc, cref, ldc );

    /* Run the reference implementation so the answers can be compared */

    REF_MMult( m, n, k, a, lda, b, ldb, cref, ldc );

    /* check output */
    for ( rep=0; rep<2; rep++ ){
      copy_matrix( m, n, cold, ldc, c, ldc );
      MY_MMult( m, n, k, a, lda, b, ldb, c, ldc );
    }

    diff = compare_matrices( m, n, c, ldc, cref, ldc );
    if(diff > 0.05f || diff < -0.05f){
        exit(0);
    }

    MY_MMult( m, n, k, a, lda, b, ldb, c, ldc );
    dtime = dclock();
    for ( rep=0; rep<NREPEATS; rep++ ){
      MY_MMult( m, n, k, a, lda, b, ldb, c, ldc );
    }

    dtime = dclock() - dtime;
    dtime /= NREPEATS;

    printf( "%d, %.3f\n", p, gflops / dtime);
    fflush( stdout );

    free( a );
    free( b );
    free( c );
    free( cold );
    free( cref );
  }

  printf( "\n" );

  exit( 0 );
}

