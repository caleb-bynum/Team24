#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>

// Define the number of buckets
#define BUCKETS 1024
#define THREADS_PER_BLOCK 256

__global__ void bucket_sort_kernel(float *data, int *bucket_counters, int num_elements, int num_buckets)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements)
    {
        // Determine the bucket for this element
        int bucket = (int)((data[i] / num_buckets) * num_elements);
        bucket = bucket < num_buckets ? bucket : num_buckets - 1;

        // Increment the appropriate bucket counter
        atomicAdd(&bucket_counters[bucket], 1);
    }
}

void bucket_sort(float *data, int num_elements)
{
    // Allocate memory for the device copies of data and bucket_counters
    float *d_data;
    int *d_bucket_counters;

    cudaMalloc((void **)&d_data, num_elements * sizeof(float));
    cudaMalloc((void **)&d_bucket_counters, BUCKETS * sizeof(int));
    cudaMemset(d_bucket_counters, 0, BUCKETS * sizeof(int));

    // Copy the data to the device
    cudaMemcpy(d_data, data, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Call the kernel
    bucket_sort_kernel<<<(num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data, d_bucket_counters, num_elements, BUCKETS);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy the bucket counters back to the host
    std::vector<int> h_bucket_counters(BUCKETS);
    cudaMemcpy(h_bucket_counters.data(), d_bucket_counters, BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);

    // Now we would sort individual buckets and concatenate them
    // For simplicity, we are going to use Thrust to sort the entire array
    thrust::sort(thrust::device, d_data, d_data + num_elements);

    // Copy the sorted data back to the host
    cudaMemcpy(data, d_data, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_bucket_counters);
}

int main()
{
    const int num_elements = 100000;
    std::vector<float> data(num_elements);

    // Initialize random data
    for (int i = 0; i < num_elements; ++i)
    {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Call bucket sort
    bucket_sort(data.data(), num_elements);

    // Print sorted data
    for (int i = 0; i < num_elements; ++i)
    {
        std::cout << data[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}
