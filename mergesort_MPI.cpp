#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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
    if (n <= 1) return;

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
    int n, rank, size;
    int *data = NULL, *local_data = NULL;
    int local_n;
    int *other_data = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assume that size is a power of 2
    if (rank == 0) {
        // Let's assume n is 16 for this example
        n = 16;
        //data = malloc(n * sizeof(int));
	data = new int[n];

        for (int i = 0; i < n; i++) {
            data[i] = rand() % 100;
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_n = n / size;
    //local_data = malloc(local_n * sizeof(int));
        local_data = new int[local_n];


    MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    merge_sort(local_data, local_n);

    // Iteratively merge data
    for (int step = 1; step < size; step = step * 2) {
        if (rank % (2 * step) != 0) {
            MPI_Send(local_data, local_n, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            break;
        }
        if (rank + step < size) {
            //other_data = malloc(local_n * sizeof(int));
	        other_data = new int[local_n];

            MPI_Recv(other_data, local_n, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            //int *temp_data = malloc(2 * local_n * sizeof(int));
	        int *temp_data = new int[2 * local_n];

            merge(temp_data, local_data, local_n, other_data, local_n);

            local_n = 2 * local_n;
            //free(local_data);
            //free(other_data);
	        delete[] local_data;
                        delete[] other_data;
            local_data = temp_data;
        }
    }

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            printf("%d ", local_data[i]);
        }
        printf("\n");
    }

    if (rank == 0) {
        //free(data);
	delete[] data;
	
    }
    //free(local_data);
        delete[] local_data;
    MPI_Finalize();
    return 0;
}
