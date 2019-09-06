#include <stdlib.h>

#define A( i,j ) a[ (i)*lda + (j) ]

void random_matrix( int m, int n, float *a, int lda, int random )
{
  double drand48();
  int i,j;

  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
    if(random==1)
      A( i,j ) = 2.0 * (float)drand48( ) - 1.0;
    else if(random>1)
      A( i, j) = i % random;
    else 
      A( i, j) = 1;
}
