#include <stdlib.h>

#define A( i,j ) a[ (i)*lda + (j) ]

void random_matrix( int m, int n, int8_t *a, int lda, int random )
{
  int i,j;

  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
    if(random==1)
      A( i,j ) = rand() % 256;
    else if(random>1)
      A( i, j) = i % random % 256;
    else 
      A( i, j) = 1;
}
