#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

// Merge two sorted sub-arrays
void merge(int *array, int *left, int left_count, int *right, int right_count) {
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
void merge_sort(int *array, int n) {
    if (n <= 1) {
       return;
    }


    int mid = n / 2;
    int *left = (int *) malloc(mid * sizeof(int));
    int *right = (int *) malloc((n - mid) * sizeof(int));

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
      printf("Number of command-line arguments: %d\n", argc);
      printf("Number of command-line arguments: %d\n", argv[0]);
    CALI_MARK_BEGIN("main");
    int n, rank, size;
    int *data = NULL, *local_data = NULL;
    int local_n;
    int *other_data = NULL;
        int sort_check = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
        printf("Total number of MPI processes (workers): %d\n", size);

    // Assume that size is a power of 2
    if (rank == 0) {
                CALI_MARK_BEGIN("data_init");
        // Let's assume n is 16 for this example
        n = 16;
        //data = malloc(n * sizeof(int));
	data = new int[n];

        for (int i = 0; i < n; i++) {
            data[i] = rand() % 1000;
        }
                CALI_MARK_END("data_init");

	// Print the initial unsorted array
        printf("Initial array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
    }

        CALI_MARK_BEGIN("comm_broad_small");
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        CALI_MARK_END("comm_broad_small");

    local_n = n / size;
    //local_data = malloc(local_n * sizeof(int));
    local_data = new int[local_n];

        CALI_MARK_BEGIN("comm_scatter_small");
    MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);
        CALI_MARK_END("comm_scatter_small");

    CALI_MARK_BEGIN("comp_sort_large");
    merge_sort(local_data, local_n);
    CALI_MARK_END("comp_sort_large");

    // Iteratively merge data
    for (int step = 1; step < size; step = step * 2) {
        if (rank % (2 * step) != 0) {
                        CALI_MARK_BEGIN("comm_send_small");
            MPI_Send(local_data, local_n, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
                        CALI_MARK_END("comm_send_small");
            break;
        }
        if (rank + step < size) {
            //other_data = malloc(local_n * sizeof(int));
	    other_data = new int[local_n];

                        CALI_MARK_BEGIN("comm_recv_small");
            MPI_Recv(other_data, local_n, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        CALI_MARK_END("comm_recv_small");
            //int *temp_data = malloc(2 * local_n * sizeof(int));
	    int *temp_data = new int[2 * local_n];

	        CALI_MARK_BEGIN("comp_merge_sort_large");
            merge(temp_data, local_data, local_n, other_data, local_n);
                        CALI_MARK_END("comp_merge_sort_large");
            local_n = 2 * local_n;
            //free(local_data);
            //free(other_data);
	    delete[] local_data;
            delete[] other_data;
            local_data = temp_data;
        }
    }

    if (rank == 0) {
	
	// Print the sorted array
        printf("Sorted array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", local_data[i]);
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
        //free(data);
	delete[] data;
	
    }
    //free(local_data);
    delete[] local_data;

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
    CALI_MARK_END("main");
    MPI_Finalize();
    return 0;
}
