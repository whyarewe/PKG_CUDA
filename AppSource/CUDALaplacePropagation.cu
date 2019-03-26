#include <cstdlib>

#include "CUDALaplacePropagation.h"
#include "CUDAUtils.h"

using namespace CUDAHelpers;

auto CUDAPropagation::laplace(ComputingData data, const Device device) -> void
{
	if (Device::GPU == device)
	{
		//laplace_gpu(data.vec,data.xAxisBound, data.yAxisBound)
	}
	else if (Device::CPU == device)
	{
		laplace_cpu(data.board, data.x_axis_bound, data.y_axis_bound, data.swarm);
	}
}

__global__ void kernel(float *data, float *out_data, const int x_axis_bound, const int y_axis_bound, const int x_heater_pos, const int y_heater_pos)
{
	const uint16_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint16_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	const auto gid = idy * y_axis_bound + idx;

	if (idx > x_axis_bound || idy > y_axis_bound)
	{
		return;
	}

	if (idy != x_heater_pos || idx != y_heater_pos)
	{
		out_data[gid] = 0.25f * (data[gid - 1] + data[gid + 1] + data[gid + y_axis_bound] + data[gid - y_axis_bound]);
	}
}

auto CUDAPropagation::laplace_cpu(std::vector<float>& vec, const int x_axis_bound, const int y_axis_bound,
                                  Entity::EntityContainer swarm) -> void
{
	auto is_under_entity = false;

	for (auto i = 1; i < y_axis_bound - 1; ++i)
	{
		for (auto j = 1; j < x_axis_bound - 1; j++)
		{
			for (const auto& entity : swarm) {
				const auto left_border = entity.getCoordinates().getX() - entity.getRadius();
				const auto right_border = entity.getCoordinates().getX() + entity.getRadius();
				const auto top_border = entity.getCoordinates().getY() - entity.getRadius();
				const auto bottom_border = entity.getCoordinates().getY() + entity.getRadius();

				if (i >= top_border && i <= bottom_border && j >= right_border && j <= left_border)
				{
					is_under_entity = true;
				}
			}

			if (!is_under_entity)
			{
				vec[i*x_axis_bound + j] = (0.25f * (vec[i*x_axis_bound + j - 1] + vec[i*x_axis_bound + j + 1]
					+ vec[i*x_axis_bound + j + y_axis_bound] + vec[i*x_axis_bound + j - y_axis_bound]));
			}
		}
	}
}

auto CUDAPropagation::laplace_gpu(std::vector<float>& vec, const int x_axis_bound, const int y_axis_bound,
                                  const int x_heater_pos, const int y_heater_pos) -> void
{
	float *data = nullptr;
	float *out_data = nullptr;
	VALID(cudaMalloc(&data, x_axis_bound * y_axis_bound * sizeof(float)));
	VALID(cudaMalloc(&out_data, x_axis_bound * y_axis_bound * sizeof(float)));

	VALID(cudaMemcpy(data, vec.data(), x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(16, 16);
	dim3 grid(38, 38);

	kernel << <grid, block >> > (data, out_data, x_axis_bound, y_axis_bound, x_heater_pos, y_heater_pos);

	VALID(cudaMemcpy(vec.data(), out_data, x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(data);
	cudaFree(out_data);
}

