#include "CUDALaplacePropagation.h"
#include "CUDAUtils.h"

using namespace CUDAHelpers;

__global__ void kernel(float *data, float *outData, int xAxisBound, int yAxisBound, int xHeaterPos, int yHeaterPos)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int idy = blockIdx.y * blockDim.y + threadIdx.y;

	const int gid = idy * yAxisBound + idx;

	if (idx > xAxisBound || idy > yAxisBound)
	{
		return;
	}

	if (idy != xHeaterPos || idx != yHeaterPos)
	{
		outData[gid] = 0.25f * (data[gid - 1] + data[gid + 1] + data[gid + yAxisBound] + data[gid - yAxisBound]);
	}

	//__syncthreads();
}

void CUDALaplacePropagation::propagate(std::vector<float> &vec, int xAxisBound, int yAxisBound, int xHeaterPos, int yHeaterPos)
{
	float *data = nullptr;
	float *outData = nullptr;
	VALID(cudaMalloc(&data, xAxisBound * yAxisBound * sizeof(float)));
	VALID(cudaMalloc(&outData, xAxisBound * yAxisBound * sizeof(float)));

	VALID(cudaMemcpy(data, &vec[0], xAxisBound * yAxisBound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(16, 16);
	dim3 grid(38, 38);

	kernel << <grid, block >> > (data, outData, xAxisBound, yAxisBound, xHeaterPos, yHeaterPos);

	VALID(cudaMemcpy(&vec[0], outData, xAxisBound * yAxisBound * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(data);
	cudaFree(outData);
}

