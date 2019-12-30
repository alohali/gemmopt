#include <stdlib.h>
#include <sys/time.h>
#define A( i,j ) a[ (i)*lda + (j) ]
void random_matrix( int m, int n, int8_t *a, int lda, int random )
{
  int i,j;
     struct timeval tv;
   gettimeofday(&tv, NULL); 

  srand(tv.tv_usec);

  for ( i=0; i<m; i++ )
    for ( j=0; j<n; j++ )
    if(random==1)
      A( i,j ) = rand() % 16-8;
    else if(random>1)
      A( i, j) = (i  + j) % random;
    else 
      A( i, j) = 1;
}
