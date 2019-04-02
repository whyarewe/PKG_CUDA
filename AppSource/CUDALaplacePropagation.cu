#include <cstdlib>

#include "CUDALaplacePropagation.h"
#include "CUDAUtils.h"
#include "Config.h"

using namespace CUDAHelpers;

auto CUDAPropagation::laplace(float* in, float* out, float* host_data, const ComputingData& data, const Device device) -> void
{
	switch (device)
	{
	case Device::GPU:
		laplace_gpu(in, out, host_data, data.x_axis_bound, data.y_axis_bound, data.board);
		break;
	case Device::CPU:
		laplace_cpu(data.board, data.x_axis_bound, data.y_axis_bound);
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

auto CUDAPropagation::laplace_cpu(std::vector<float>& vec, const uint32_t x_axis_bound,
                                  const uint32_t y_axis_bound) -> void
{
	std::vector<float> out_vec(vec.capacity());

	for (auto i = 1u; i < y_axis_bound - 1; ++i)
	{
		for (auto j = 1u; j < x_axis_bound - 1; j++)
		{
			out_vec[i * x_axis_bound + j] = (0.25f * (vec[i * x_axis_bound + j - 1] + vec[i * x_axis_bound + j + 1]
				+ vec[i * x_axis_bound + j + y_axis_bound] + vec[i * x_axis_bound + j - y_axis_bound]));
		}
	}

	vec = out_vec;
}

auto CUDAPropagation::laplace_gpu(float* data, float* out_data, float* host_data, uint32_t x_axis_bound,
	uint32_t y_axis_bound, std::vector<float>& vec) -> void
{
	host_data = vec.data();

	VALID(cudaMemcpyAsync(data, host_data, x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid(Config::StandardResolution::width / block.x, Config::StandardResolution::height / block.y);

	kernel << <grid, block >> > (data, out_data, x_axis_bound, y_axis_bound);

	VALID(cudaMemcpyAsync(vec.data(), out_data, x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyDeviceToHost));
}