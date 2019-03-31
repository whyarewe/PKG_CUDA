#include <cstdlib>

#include "CUDALaplacePropagation.h"
#include "CUDAUtils.h"

using namespace CUDAHelpers;

auto CUDAPropagation::laplace(ComputingData data, const Device device) -> void
{
	switch (device)
	{
	case Device::GPU:
		laplace_gpu(data.board, data.x_axis_bound, data.y_axis_bound);
		break;
	case Device::CPU:
		laplace_cpu(data.board, data.x_axis_bound, data.y_axis_bound, data.swarm);
		break;
	default:
		std::cerr << "CUDA Progatation: Critical error, unknown device!" << std::endl;
		std::exit(0);
	}
}

__global__ void kernel(float* data, float* out_data, const int x_axis_bound, const int y_axis_bound)
{
	const uint16_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint16_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	const auto gid = idy * y_axis_bound + idx;

	if (idx > x_axis_bound || idy > y_axis_bound)
	{
		return;
	}

	out_data[gid] = 0.25f * (data[gid - 1] + data[gid + 1] + data[gid + y_axis_bound] + data[gid - y_axis_bound]);
}

auto CUDAPropagation::laplace_cpu(std::vector<float>& vec, const uint32_t x_axis_bound, const uint32_t y_axis_bound,
                                  Entity::EntityContainer swarm) -> void
{
	auto is_under_entity = false;

	for (auto i = 1u; i < y_axis_bound - 1; ++i)
	{
		for (auto j = 1u; j < x_axis_bound - 1; j++)
		{
			for (const auto& entity : swarm)
			{
				if (i >= entity.getDimensions().getTopBorder()
					&& i <= entity.getDimensions().getBottomBorder()
					&& j >= entity.getDimensions().getRightBorder()
					&& j <= entity.getDimensions().getLeftBorder())
				{
					is_under_entity = true;
				}
			}

			if (!is_under_entity)
			{
				vec[i * x_axis_bound + j] = (0.25f * (vec[i * x_axis_bound + j - 1] + vec[i * x_axis_bound + j + 1]
					+ vec[i * x_axis_bound + j + y_axis_bound] + vec[i * x_axis_bound + j - y_axis_bound]));
			}
		}
	}
}

auto CUDAPropagation::laplace_gpu(std::vector<float>& vec, const uint32_t x_axis_bound, const uint32_t y_axis_bound) -> void
{
	float* data = nullptr;
	float* out_data = nullptr;
	VALID(cudaMalloc(&data, x_axis_bound * y_axis_bound * sizeof(float)));
	VALID(cudaMalloc(&out_data, x_axis_bound * y_axis_bound * sizeof(float)));

	VALID(cudaMemcpy(data, vec.data(), x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(10, 10);
	dim3 grid(60, 60);

	kernel << <grid, block >> >(data, out_data, x_axis_bound, y_axis_bound);

	VALID(cudaMemcpy(vec.data(), out_data, x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(data);
	cudaFree(out_data);
}
