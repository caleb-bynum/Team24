#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <ctime>

// Merge two sorted sub-arrays
void merge(float *array, float *left, int left_count, float *right, int right_count) {
    int i = 0, j = 0, k = 0;
    while (i < left_count && j < right_count) {
        if (left[i] < right[j]) {
            array[k++] = left[i++];
        } else {
            array[k++] = right[j++];
        }
    }
    while (i < left_count) {
        array[k++] = left[i++];
    }
    while (j < right_count) {
        array[k++] = right[j++];
    }
}

// Serial merge sort
void merge_sort(float *array, int n) {
    if (n <= 1) {
       return;
    }

    int mid = n / 2;
    float *left = (float *) malloc(mid * sizeof(float));
    float *right = (float *) malloc((n - mid) * sizeof(float));

    for (int i = 0; i < mid; i++) left[i] = array[i];
    for (int i = mid; i < n; i++) right[i - mid] = array[i];

    merge_sort(left, mid);
    merge_sort(right, n - mid);
    merge(array, left, mid, right, n - mid);

    free(left);
    free(right);
}

int main(int argc, char **argv) {
    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();
    CALI_MARK_BEGIN("main");
    int n, rank, size;
    float *data = NULL, *local_data = NULL;
    int local_n;
    float *other_data = NULL;
    int sort_check = 0;

    // amount of data from cmd line
    if (argc == 2)
    {
        n = atoi(argv[1]);
    }
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  

    // Assume that size is a power of 2
    if (rank == 0) {
        printf("Data Size: %d\n", n);
        printf("\n");
        printf("Processors: %d\n", size);
        printf("\n");
        CALI_MARK_BEGIN("data_init");
        // Let's assume n is 16 for this example
       	data = new float[n];
              srand(time(NULL)); // Seed the RNG with the current time
        for (int i = 0; i < n; i++) {
            data[i] = static_cast<float>(rand()) / RAND_MAX * n; // random
            //data[i] = static_cast<float>(i); // sorted
            //data[i] = static_cast<float>(n - i - 1); // Reverse sorted values
        }        
              CALI_MARK_END("data_init");

	// Print the initial unsorted array
        // printf("Initial array: ");
        // for (int i = 0; i < n; i++) {
        //     printf("%d ", data[i]);
        // }
        // printf("\n");
    }

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Bcast");
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Bcast");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    local_n = n / size;
    local_data = new float[local_n];

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    merge_sort(local_data, local_n);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Iteratively merge data
    for (int step = 1; step < size; step = step * 2) {
        if (rank % (2 * step) != 0) {
            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_small");
	    CALI_MARK_BEGIN("MPI_Send");
            MPI_Send(local_data, local_n, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            CALI_MARK_END("MPI_Send");
	    CALI_MARK_END("comm_small");
            CALI_MARK_END("comm");
            break;
        }
        if (rank + step < size) {
            //other_data = malloc(local_n * sizeof(int));
            other_data = new float[local_n];
            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_small");
	    CALI_MARK_BEGIN("MPI_Recv");
            MPI_Recv(other_data, local_n, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CALI_MARK_END("MPI_Recv");
	    CALI_MARK_END("comm_small");
            CALI_MARK_END("comm");
            //int *temp_data = malloc(2 * local_n * sizeof(int));
	    float *temp_data = new float[2 * local_n];

            CALI_MARK_BEGIN("comp");
	    CALI_MARK_BEGIN("comp_large");
            merge(temp_data, local_data, local_n, other_data, local_n);
            CALI_MARK_END("comp_large");
            CALI_MARK_END("comp");

            local_n = 2 * local_n;
	    delete[] local_data;
            delete[] other_data;
            local_data = temp_data;
        }
    }

    if (rank == 0) {
	
                 // Print the sorted array
        //printf("Sorted array: ");
        //for (int i = 0; i < n; i++) {
            //printf("%d ", local_data[i]);
        //}
        //printf("\n");

        // Print the first 10 elements of the sorted array
        printf("First 10 elements of sorted array: ");
        for (int i = 0; i < std::min(10, n); i++) {
            printf("%f ", local_data[i]);
        }
        printf("\n");

        // Print the last 10 elements of the sorted array
        printf("Last 10 elements of sorted array: ");
        for (int i = std::max(0, n - 10); i < n; i++) {
            printf("%f ", local_data[i]);
        }
        printf("\n");

         CALI_MARK_BEGIN("correctness_check");
    // Check if data is sorted.
    for (int j = 1; j < n; j++) {
        if (local_data[j-1] > local_data[j]) {
            fprintf(stderr, "Error: data is not sorted at index %d\n", j);
	        sort_check++;
        }
    }

        if (sort_check == 0){
               printf("Array is Sorted!");
          }
        CALI_MARK_END("correctness_check");
    }

    if (rank == 0) {
	    delete[] data;
	
    }
    delete[] local_data;

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    adiak::value("num_threads", 0); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", 0); // The number of CUDA blocks 
    adiak::value("group_num", 24); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI / Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
    CALI_MARK_END("main");
    MPI_Finalize();
    return 0;
}
