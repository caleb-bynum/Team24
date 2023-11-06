#include "Sort.h" /* Odd-Even Sorting Routine */

/* function to get command line arguments*/
void Get_args( int argc, char* argv[], int my_rank, int p, MPI_Comm comm, int* global_n_p, int* local_n_p ) {
    if ( my_rank == 0 ) {
        if ( argc != 2 ) {
            fprintf( stderr, "usage: mpirun -np <number of processes> %s <number of elements>\n", argv[0] );
            fflush( stderr );
            *global_n_p = 0;
        }
        else {
            *global_n_p = strtol( argv[1], NULL, 10 );
        }
    }
    /* broadcast global array size to all processes */
    MPI_Bcast( global_n_p, 1, MPI_INT, 0, comm ); //

    /* check for invalid input */
    if ( *global_n_p <= 0 ) {
        MPI_Finalize();
        exit( -1 );
    }

    /* calculate local_n */
    *local_n_p = *global_n_p / p;
}

int main( int argc, char* argv[] ) {
    /* MPI variables */
    int p; // number of processes
    int my_rank; 
    MPI_Comm comm;

    /* local variables */
    double* local_A; // local array
    int local_n; // size of local array
    int global_n; // size of global array 

    /* initialize MPI */
    MPI_Init( &argc, &argv );
    comm = MPI_COMM_WORLD;
    MPI_Comm_size( comm, &p );
    MPI_Comm_rank( comm, &my_rank );

    /* receive command line arguments */
    Get_args( argc, argv, my_rank, p, comm, &global_n, &local_n );
    local_A = (double*) malloc( local_n * sizeof(double) );

    /* generate random values in array */
    srandom( my_rank + 1 );
    for ( int i = 0; i < local_n; i++ ) {
        local_A[i] = random() / (double) RAND_MAX;
    }

    /* start main procedure */
    MPI_Barrier( comm );
    Sort( local_A, local_n, p, my_rank, comm );
    MPI_Barrier( comm );

    /* check correctness of sorting */
    if ( my_rank == 0 ) {
        for ( int i = 1; i < p; i++ ) {
            double* temp = (double*) malloc( local_n * sizeof(double) );
            MPI_Recv( temp, local_n, MPI_DOUBLE, i, 0, comm, MPI_STATUS_IGNORE );
            for ( int j = 0; j < local_n; j++ ) {
                local_A[j + i * local_n] = temp[j];
            }
            free( temp );
        }
        for ( int i = 1; i < global_n; i++ ) {
            if ( local_A[i - 1] > local_A[i] ) {
                printf( "Error: local_A[%d] = %f > local_A[%d] = %f\n", i - 1, local_A[i - 1], i, local_A[i] );
                fflush( stdout );
                break;
            }
        }
        /* print if array sorted */
        printf( "Correctness check passed (if no preceding errors).\n" );
    }
    else {
        MPI_Send( local_A, local_n, MPI_DOUBLE, 0, 0, comm );
    }


    /* finalize program */
    free( local_A );
    MPI_Finalize();

    return 0;
}