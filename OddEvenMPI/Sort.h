#include <stdlib.h> /* STL Quicksort */
#include "OddEvenIteration.h" /* Odd_Even_Iteration */



/* Compare function required by qsort */
int Compare( const void* a_p, const void* b_p ) {
    double a = * (double*) a_p;
    double b = * (double*) b_p;

    if ( a < b ) {
        return -1;
    }
    else if ( a == b ) {
        return 0;
    }
    else {
        return 1;
    }
}

/* Given array and MPI parameters, executes OddEvenTranspositionSort */
void Sort( double* local_A, int local_n, int p, int my_rank, MPI_Comm comm ) {
    /* allocate additional local memory */
    double* temp_B = (double*) malloc( local_n * sizeof(double) );
    double* temp_C = (double*) malloc( local_n * sizeof(double) );

    int even_partner;
    int odd_partner;

    /* determine partners for even and odd phases */
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    if ( my_rank % 2 ) {
        even_partner = my_rank - 1;
        odd_partner = my_rank + 1;
        if ( odd_partner == p ) {
            odd_partner = MPI_PROC_NULL;
        }
    }
    else {
        even_partner = my_rank + 1;
        if ( even_partner == p ) {
            even_partner = MPI_PROC_NULL;
        }
        odd_partner = my_rank - 1;
    }
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    /* use built-in quicksort for each local array */
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    qsort( local_A, local_n, sizeof(double), Compare );
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    /* execute odd or even exchange procedure, depending on phase */
    for ( int phase = 0; phase < p; phase++ ) {
        Odd_Even_Iteration( local_A, temp_B, temp_C, local_n, my_rank, comm, phase, even_partner, odd_partner );
    }

    /* free additional local memory */
    free( temp_B );
    free( temp_C );

}