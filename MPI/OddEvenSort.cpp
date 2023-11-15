#include "Sort.h" /* Odd-Even Sorting Routine */

#include "mpi.h"
#include <stdio.h>
#include <limits.h>

/* fill array with random values */
void fill_array_random( double* local_A, int local_n, int my_rank ) {
    srand( my_rank + 1 );
    for ( int i = 0; i < local_n; i++ ) {
        local_A[i] = random() / (double) RAND_MAX;
    }
}

/* fill array with sorted values */
void fill_array_sorted( double* local_A, int local_n, int global_n, int my_rank ) {
    for ( int i = 0; i < local_n; i++ ) {
        local_A[i] = (((float) my_rank) + ((float) i)) / ((float) global_n);
    }
}


void fill_array_reverse_sorted( double* local_A, int local_n, int global_n, int my_rank ) {
    for ( int i = 0; i < local_n; i++ ) {
        local_A[i] = (((float) my_rank) + ((float) local_n) - ((float) i)) / ((float) global_n);
    }
}

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
    /* Initialize CALI */
    cali::ConfigManager mgr;
    mgr.start();
    CALI_MARK_BEGIN("main");

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

    /* Adiak variables */
    const char* algorithm = "OddEvenSort";
    const char* programmingModel = "MPI";
    const char* datatype = "double";
    const int sizeOfDatatype = sizeof(double);
    const int inputSize = global_n;
    const char* inputType = "Random";
    const int num_procs = p;
    const int group_number = 1; // unsure what this is
    const char* implementation_source = "Handwritten+Online";

    /* generate random values in array */
    CALI_MARK_BEGIN("data_init");
    fill_array_random( local_A, local_n, my_rank );
    CALI_MARK_END("data_init");


    /* start main procedure */
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    MPI_Barrier( comm );
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    Sort( local_A, local_n, p, my_rank, comm );

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    MPI_Barrier( comm );
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    /* initialize global array */
    double* global_A = (double*) malloc( global_n * sizeof(double));

    /* gather each processes' local_A array and place into global_A */
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    MPI_Gather( local_A, local_n, MPI_DOUBLE, global_A, local_n, MPI_DOUBLE, 0, comm );
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");
    
    /* check correctness of sorting */
    CALI_MARK_BEGIN("correctness_check");
    if ( my_rank == 0 ) {
        for ( int i = 1; i < global_n; i++ ) {
            if ( global_A[i - 1] > global_A[i] ) {
                printf( "Error: global_A[%d] = %f > global_A[%d] = %f\n", i - 1, global_A[i - 1], i, global_A[i] );
                fflush( stdout );
                break;
            }
        }
        free(global_A);
        printf( "Correctness check passed (if no preceding errors).\n");
    }
    else {
        MPI_Send( local_A, local_n, MPI_DOUBLE, 0, 0, comm );
    }
    CALI_MARK_END("correctness_check");

    /* Adiak Metadata */
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    /* finalize program */
    free( local_A );
    MPI_Finalize();
    mgr.stop();
    mgr.flush();
    CALI_MARK_END("main");

}