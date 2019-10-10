#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define abs( x ) ( (x) < 0.0 ? -(x) : (x) )

#include <stdint.h>
#include <stdio.h>

int8_t compare_matrices( int m, int n, int8_t *a, int lda, int8_t *b, int ldb )
{
//    printf("\n---result----\n");
//    print_matrix(m, n, a, lda);
//    printf("\n-------\n");
//    print_matrix(m, n, b, ldb);
//    printf("\n-------\n");
  int i, j;
  int8_t max_diff = 0, diff;
  int printed = 0;

  for ( i=0; i<m; i++ ){
    for ( j=0; j<n; j++ ){
      diff = abs( A( i,j ) - B( i,j ) );
      max_diff = ( diff > max_diff ? diff : max_diff );
      if(100 >= printed)
      if(diff > 0) {
        printf("\n error: i %d  j %d diff %d a %d, b %d\n", i, j, max_diff, A(i, j), B(i,j));
        printed += 1;
      }
    }
  }

  return max_diff;
}

