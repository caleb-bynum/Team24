#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>

#define BUCKETS 1024
#define THREADS_PER_BLOCK 256

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

    cudaMemcpy(d_data, data, elements * sizeof(float), cudaMemcpyHostToDevice);
    bucket_sort_kernel<<<(elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data, d_bucket_counters, elements, BUCKETS);
    cudaDeviceSynchronize();

    std::vector<int> h_bucket_counters(BUCKETS);
    cudaMemcpy(h_bucket_counters.data(), d_bucket_counters, BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::sort(thrust::device, d_data, d_data + elements);
    cudaMemcpy(data, d_data, elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_bucket_counters);
}

int main()
{
    const int num_elements = 100000;
    std::vector<float> data(num_elements);

    for (int i = 0; i < num_elements; ++i)
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    bucket_sort(data.data(), num_elements);

    for (int i = 0; i < num_elements; ++i)
        std::cout << data[i] << ' ';
    std::cout << std::endl;

    return 0;
}
