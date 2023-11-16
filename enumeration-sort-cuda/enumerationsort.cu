#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ void enum_sort(int *a, int *b, int *c, int N)
{
	int tid = threadIdx.x + blockDim.x *blockIdx.x;
	int count = 0;
	int d;
	for (d = 0; d < N; d++)
	{
		if (a[d] < a[tid])
		{
			count++;
		}
	}

	c[count] = a[tid];

}

int main(int argc, char *argv[])
{
	int i, j;
	//CUDA Grid structure values	
	int Grid_Dim_x = 1, Grid_Dim_y = 1;	
	int Block_Dim_x = 1, Block_Dim_y = 1;	
	int noThreads_x, noThreads_y;	
	int noThreads_block;	
	//array length
	int N = 10;	
	int B;
	int T;
	int *a, *b, *c, *d;
	int *dev_a, *dev_b, *dev_c;
	int size;	
	// using cuda events to get time
	cudaEvent_t start, stop;	
	float elapsed_time_ms;	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	//input
	printf("Enter the value for N: ");
	scanf("%d", &N);
	//takes in input
	int valid = 0;
	while (valid == 0)
	{
		printf("Enter the number of blocks: ");
		scanf("%d", &B);

		printf("Enter the number of threads: ");
		scanf("%d", &T);

		if (B > 1024 || T > 1024 || B * T < N)
		{
			printf("Invlaid input entered.\n");
		}
		else
		{
			valid = 1;
			Grid_Dim_x = B;
			Block_Dim_x = T;	//puts the size of blocks and thread in for the dim3
		}
	}

	dim3 Grid(Grid_Dim_x, Grid_Dim_x);	
	dim3 Block(Block_Dim_x, Block_Dim_y);	
	//size of array in bytes
	size = N *N* sizeof(int);	

	a = (int*) malloc(size);	
	b = (int*) malloc(size);
	c = (int*) malloc(size);	
	d = (int*) malloc(size);	
	
	//random seed initialize
	srand(3);	

	for (i = 0; i < N; i++)
	{
		//populate numbers
		a[i] = (int) rand();
	}

	cudaMalloc((void **) &dev_a, size);	
	cudaMalloc((void **) &dev_b, size);
	cudaMalloc((void **) &dev_c, size);

	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

	//start
	cudaEventRecord(start, 0);	

	enum_sort <<<Grid, Block>>> (dev_a, dev_b, dev_c, N);
	cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

	// end 
	cudaEventRecord(stop, 0);	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);

	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);
	double gpuTime = elapsed_time_ms;

	//check cpu

	cudaEventRecord(start, 0);	// use same timing*

	//cpu_matrixmult(a,b,d,N);				// do calculation on host
	//sequential rank sort
	int k;
	for (k = 0; k < N; k++)
	{
		int count = 0;
		int d;
		for (d = 0; d < N; d++)
		{
			if (a[d] < a[k])
			{
				count++;
			}
		}

		b[count] = a[k];
		count = 0;
	}

	cudaEventRecord(stop, 0);	// measure end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);

	printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms);	// exe. time
	double cpuTime = elapsed_time_ms;

	//compare results

	printf("Initial Array: \n");
	int h;
	for (h = 0; h < N; h++)
	{
		printf("%d ", a[h]);
	}

	printf("\n");

	printf("Sequential Enum Sort: \n");

	for (k = 0; k < N; k++)
	{
		int count = 0;
		int d;
		for (d = 0; d < N; d++)
		{
			if (a[d] < a[k])
			{
				count++;
			}
		}

		b[count] = a[k];
		count = 0;
	}

	for (h = 0; h < N; h++)
	{
		printf("%d ", b[h]);
	}

	int error = 0;
	int r;
	for (r = 0; r < N; r++)
	{
		if (b[r] != c[r])
		{
			error = 1;
			break;
		}
	}


	//free memmory
	free(a);
	free(b);
	free(c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}