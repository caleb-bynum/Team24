// #include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int BUCKETS;
int THREADS_PER_BLOCK;
const char *data_init = "data_init";
const char *comm = "comm";
const char *cudaMemcpy_region = "cudaMemcpy";
const char *correctness_check = "correctness_check";

__global__ void bucket_sort_kernel(float *data, int *bucket_counters, int elements, int buckets)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements)
    {
        int bucket = (int)((data[i] / buckets) * elements);
        bucket = bucket < buckets ? bucket : buckets - 1;
        atomicAdd(&bucket_counters[bucket], 1);
    }
}

void bucket_sort(float *data, int elements)
{
    float *d_data;
    int *d_bucket_counters;

    cudaMalloc((void **)&d_data, elements * sizeof(float));
    cudaMalloc((void **)&d_bucket_counters, BUCKETS * sizeof(int));
    cudaMemset(d_bucket_counters, 0, BUCKETS * sizeof(int));

    // CALI_MARK_BEGIN(comm);
    cudaMemcpy(d_data, data, elements * sizeof(float), cudaMemcpyHostToDevice);
    // CALI_MARK_END(comm);

    bucket_sort_kernel<<<(elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data, d_bucket_counters, elements, BUCKETS);
    cudaDeviceSynchronize();

    std::vector<int> h_bucket_counters(BUCKETS);

    // CALI_MARK_BEGIN(cudaMemcpy_region);
    cudaMemcpy(h_bucket_counters.data(), d_bucket_counters, BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);
    // CALI_MARK_END(cudaMemcpy_region);

    thrust::sort(thrust::device, d_data, d_data + elements);
    // CALI_MARK_BEGIN(cudaMemcpy_region);
    cudaMemcpy(data, d_data, elements * sizeof(float), cudaMemcpyDeviceToHost);
    // CALI_MARK_END(cudaMemcpy_region);

    cudaFree(d_data);
    cudaFree(d_bucket_counters);
}

int main(int argc, char *argv[])
{
    cali::ConfigManager mgr;
    mgr.start();

    BUCKETS = atoi(argv[1]);
    const int num_elements = atoi(argv[2]);
    THREADS_PER_BLOCK = BUCKETS / 4;

    std::vector<float> data(num_elements);

    CALI_MARK_BEGIN(data_init);
    for (int i = 0; i < num_elements; ++i)
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    CALI_MARK_END(data_init);

    bucket_sort(data.data(), num_elements);

    for (int i = 0; i < num_elements; ++i)
        std::cout << data[i] << ' ';
    std::cout << std::endl;

    adiak::init(NULL);
    adiak::launchdate();                           // launch date of the job
    adiak::libraries();                            // Libraries used
    adiak::cmdline();                              // Command line used to launch the job
    adiak::clustername();                          // Name of the cluster
    adiak::value("Algorithm", "SampleSort");       // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");      // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float");             // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", num_elements);       // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random");
    adiak::value("num_threads", BUCKETS);                                           // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", THREADS_PER_BLOCK);                                  // The number of CUDA blocks
    adiak::value("group_num", 24);                                                  // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten with some reference to AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
}
