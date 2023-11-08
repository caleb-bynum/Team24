#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>

int compare(const void* first, const void* second){
    double diff = *(double*)first - *(double*)second;

    if (diff > 0)
        return 1;
    else if (diff < 0)
        return -1;

    return 0;
}

int find_bucket(double input, int processors){
    for(int i = 1; i < processors + 1; i++){
        double limit = (double)i / (double)processors;
        if(input <= limit)
            return i - 1;
    }
}

void custom_qsort(double *input, int input_size){
    qsort(input, (size_t)input_size, sizeof(double), compare);
    return;
} 


int main(int argc, char *argv[]){
    int rank, P;
    int bucket_size[1];
    int N = strtol(argv[1], NULL, 10); // N numbers to be sorted
    int *bucket_num = (int*)malloc(P*sizeof(int));
    double *pivots = (double*)malloc(N*sizeof(double));
    int *displacements = (int*)malloc(P*sizeof(int));
    double *d_list = (double*)malloc(N*sizeof(double));
    int *index = (int*)malloc(P*sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    double start_time = MPI_Wtime();

    if(rank == 0){
        for(int i = 0; i < N; i++){
            pivots[i] = (double)rand() / (double) RAND_MAX;
            bucket_num[find_bucket(pivots[i], P)]++;
        }
    }

    MPI_Scatter(bucket_num, 1, MPI_INT, bucket_size, 1, MPI_INT, 0, MPI_COMM_WORLD );
    double *bucket_list = (double*)malloc(bucket_size[0]*sizeof(double));

    if(rank == 0){
        displacements[0] = 0;
        for(int j = 1; j < P; j++)
            displacements[j] = bucket_num[j-1] + displacements[j-1];

        int bucket;
        for(int j = 0; j < N; j++){
            bucket = find_bucket(pivots[j], P);
            d_list[displacements[bucket] + index[bucket]] = pivots[j];
            index[bucket]++;
        }
        free(pivots);
        free(index);
    }

    MPI_Scatterv(d_list, bucket_num, displacements, MPI_DOUBLE, bucket_list, bucket_size[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    custom_qsort(bucket_list, bucket_size[0]);
    MPI_Gatherv(bucket_list, bucket_size[0], MPI_DOUBLE, d_list, bucket_num, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    double end_time = MPI_Wtime();

    free(bucket_list);

    if(rank == 0){
        free(displacements);
        free(bucket_num);
        free(d_list);
        printf("\nTotal Execution Time: %f\n", end_time - start_time);
    }
    MPI_Finalize();
}