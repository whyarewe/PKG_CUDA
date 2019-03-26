#include <cstdlib>

#include "CUDALaplacePropagation.h"
#include "CUDAUtils.h"

using namespace CUDAHelpers;

__global__ void kernel(float *data, float *outData, int xAxisBound, int yAxisBound, int xHeaterPos, int yHeaterPos)
{
	const uint16_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint16_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	const int gid = idy * yAxisBound + idx;

	if (idx > xAxisBound || idy > yAxisBound)
	{
		return;
	}

	if (idy != xHeaterPos || idx != yHeaterPos)
	{
		outData[gid] = 0.25f * (data[gid - 1] + data[gid + 1] + data[gid + yAxisBound] + data[gid - yAxisBound]);
	}
}

void CUDAPropagation::laplace(ComputingData data, Device device)
{
	if (Device::GPU == device)
	{
		//laplace_gpu(data.vec,data.xAxisBound, data.yAxisBound)
	}
	else if (Device::CPU == device)
	{
		laplace_cpu(data.board, data.xAxisBound, data.yAxisBound, data.swarm);
	}
}

void CUDAPropagation::laplace_cpu(std::vector<float>& vec, int xAxisBound, int yAxisBound, Entity::EntityContainer swarm)
{
	bool isUnderEntity = false;

	for (int i = 1; i < yAxisBound - 1; ++i)
	{
		for (int j = 1; j < xAxisBound - 1; j++)
		{
			for (const auto& entity : swarm) {
				uint32_t leftBorder = entity.getCoordinates().getX() - entity.getRadius();
				uint32_t rightBorder = entity.getCoordinates().getX() + entity.getRadius();
				uint32_t topBorder = entity.getCoordinates().getY() - entity.getRadius();
				uint32_t bottomBorder = entity.getCoordinates().getY() + entity.getRadius();

				if (i >= topBorder && i <= bottomBorder && j >= rightBorder && j <= leftBorder)
				{
					isUnderEntity = true;
				}
			}

			if (!isUnderEntity)
			{
				vec[i*xAxisBound + j] = (0.25f * (vec[i*xAxisBound + j - 1] + vec[i*xAxisBound + j + 1]
					+ vec[i*xAxisBound + j + yAxisBound] + vec[i*xAxisBound + j - yAxisBound]));
			}
		}
	}
}

void CUDAPropagation::laplace_gpu(std::vector<float> &vec, int xAxisBound, int yAxisBound, int xHeaterPos, int yHeaterPos)
{
	float *data = nullptr;
	float *outData = nullptr;
	VALID(cudaMalloc(&data, xAxisBound * yAxisBound * sizeof(float)));
	VALID(cudaMalloc(&outData, xAxisBound * yAxisBound * sizeof(float)));

	VALID(cudaMemcpy(data, vec.data(), xAxisBound * yAxisBound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(16, 16);
	dim3 grid(38, 38);

	kernel << <grid, block >> > (data, outData, xAxisBound, yAxisBound, xHeaterPos, yHeaterPos);

	VALID(cudaMemcpy(vec.data(), outData, xAxisBound * yAxisBound * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(data);
	cudaFree(outData);
}

