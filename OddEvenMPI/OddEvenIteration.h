#include "Merge.h" /* Merge_high, Merge_low */
#include "mpi.h" /* MPI API */

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

/* Based on the current phase, use MPI to exchange elements with partner process. Merge these elements locally */
void Odd_Even_Iteration( double* local_A, double* temp_B, double* temp_C, int local_n, int my_rank, MPI_Comm comm, int phase, int even_partner, int odd_partner ) {
    MPI_Status status;

    /* if phase is even */
    if ( phase % 2 == 0 ) {
        /* if even partner exists */
        if ( even_partner >= 0 ) {

            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            MPI_Sendrecv( local_A, local_n, MPI_DOUBLE, even_partner, 0, temp_B, local_n, MPI_DOUBLE, even_partner, 0, comm, &status );
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");

            /* if my rank is odd */
            if ( my_rank % 2 ) {
                CALI_MARK_BEGIN("comp");
                CALI_MARK_BEGIN("comp_large");
                Merge_High( local_A, temp_B, temp_C, local_n );
                CALI_MARK_END("comp_large");
                CALI_MARK_END("comp");
            }
            /* if my rank is even */
            else {
                CALI_MARK_BEGIN("comp");
                CALI_MARK_BEGIN("comp_large");
                Merge_Low( local_A, temp_B, temp_C, local_n );
                CALI_MARK_END("comp_large");
                CALI_MARK_END("comp");
            }
        }
    }
    /* if phase is odd */
    else {
        /* if odd partner exists */
        if ( odd_partner >= 0 ) {

            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            MPI_Sendrecv( local_A, local_n, MPI_DOUBLE, odd_partner, 0, temp_B, local_n, MPI_DOUBLE, odd_partner, 0, comm, &status );
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");

            /* if my rank is odd */
            if ( my_rank % 2 ) {
                CALI_MARK_BEGIN("comp");
                CALI_MARK_BEGIN("comp_large");
                Merge_Low( local_A, temp_B, temp_C, local_n );
                CALI_MARK_END("comp_large");
                CALI_MARK_END("comp");
            }
            /* if my rank is even */
            else {
                CALI_MARK_BEGIN("comp");
                CALI_MARK_BEGIN("comp_large");
                Merge_High( local_A, temp_B, temp_C, local_n );
                CALI_MARK_END("comp_large");
                CALI_MARK_END("comp");
            }
        }
    }
}