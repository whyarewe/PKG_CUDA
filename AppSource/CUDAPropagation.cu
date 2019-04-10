#include <cstdlib>

#include "CUDAPropagation.h"
#include "CUDAUtils.h"
#include "Config.h"

using namespace CUDAHelpers;

auto CUDAPropagation::propagate(float* in, float* out, const ComputingData& data, const Device device, const Method method) -> void
{
	switch (device)
	{
	case Device::GPU:
		propagate_gpu(in, out, data, method);
		break;
	case Device::CPU:
		propagate_cpu(data, method);
		break;
	default:
		std::cerr << "CUDA Propagation: Critical error, unknown device!" << std::endl;
		std::exit(0);
	}
}

auto CUDAPropagation::propagate_cpu(const ComputingData& data, const Method method) -> void
{
	switch (method)
	{
	case Method::Laplace:
		laplace_cpu(data.board, data.x_axis_bound, data.y_axis_bound);
		break;
	case Method::FTCS:
		ftcs_cpu(data.board, data.x_axis_bound, data.y_axis_bound);
		break;
	default:
		std::cerr << "CUDA Propagation: Critical error, unknown method!" << std::endl;
		std::exit(0);
	}
}

auto CUDAPropagation::propagate_gpu(float* in, float* out, const ComputingData& data, const Method method) -> void
{
	switch (method)
	{
	case Method::Laplace:
		laplace_gpu(in, out, data.x_axis_bound, data.y_axis_bound, data.board);
		break;
	case Method::FTCS:
		ftcs_gpu(in, out, data.x_axis_bound, data.y_axis_bound, data.board);
		break;
	default:
		std::cerr << "CUDA Propagation: Critical error, unknown method!" << std::endl;
		std::exit(0);
	}
}

__global__ void kernel_laplace(float* data, float* out_data, const int x_axis_bound, const int y_axis_bound)
{
	const uint16_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint16_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	const auto gid = idy * x_axis_bound + idx;

	if (idx > x_axis_bound || idy > y_axis_bound)
	{
		return;
	}

	out_data[gid] = 0.25f * (data[gid - 1] + data[gid + 1] + data[gid + x_axis_bound] + data[gid - x_axis_bound]);
}

__global__ void kernel_ftcs(float* data, float* out_data, const int x_axis_bound, const int y_axis_bound, const float r, const float r2)
{
	const uint16_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint16_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	const auto gid = idy * x_axis_bound + idx;

	if (idx > x_axis_bound || idy > y_axis_bound)
	{
		return;
	}

	float horizontal = 0.25 * r2 * (data[gid - 1] + data[gid + 1] +
		data[gid + x_axis_bound] + data[gid - x_axis_bound]);

	float diagonal = 0.25 * r * (data[gid - x_axis_bound - 1] + data[gid - x_axis_bound + 1] +
		data[gid + x_axis_bound - 1] + data[gid + x_axis_bound + 1]);

	out_data[gid] = horizontal + diagonal;
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
				+ vec[(i + 1) * x_axis_bound + j] + vec[(i - 1) * x_axis_bound + j]));
		}
	}

	vec = out_vec;
}

auto CUDAPropagation::ftcs_cpu(std::vector<float>& vec, const uint32_t x_axis_bound, const uint32_t y_axis_bound) -> void
{
	std::vector<float> out_vec(vec.capacity());

	const float dx = 1;
	const float dt = 1.f / 60.f;
	const float alpha = 0;
	const float r = (alpha * dt) / (dx * dx);
	const float r2 = 1 - 2 * r;

	for (auto i = 1u; i < y_axis_bound - 1; ++i)
	{
		for (auto j = 1u; j < x_axis_bound - 1; j++)
		{
			float horizontal = 0.25 * r2 * (vec[(i - 1) * x_axis_bound + j] + vec[(i + 1) * x_axis_bound + j] +
				vec[i * x_axis_bound + j - 1] + vec[i * x_axis_bound + j + 1]);

			float diagonal = 0.25 * r * (vec[(i - 1) * x_axis_bound + j - 1] + vec[(i - 1) * x_axis_bound + j + 1] +
				vec[(i + 1) * x_axis_bound + j - 1] + vec[(i + 1) * x_axis_bound + j - 1]);

			out_vec[i * x_axis_bound + j] = horizontal + diagonal;
		}
	}

	vec = out_vec;
}
auto CUDAPropagation::laplace_gpu(float* data, float* out_data, const uint32_t x_axis_bound,
	const uint32_t y_axis_bound, std::vector<float>& vec) -> void
{
	VALID(cudaMemcpyAsync(data, vec.data(), x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid(x_axis_bound / block.x, y_axis_bound / block.y);

	kernel_laplace << <grid, block >> > (data, out_data, x_axis_bound, y_axis_bound);

	VALID(cudaMemcpyAsync(vec.data(), out_data, x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyDeviceToHost));
}

auto CUDAPropagation::ftcs_gpu(float* data, float* out_data, const uint32_t x_axis_bound,
	const uint32_t y_axis_bound, std::vector<float>& vec) -> void
{
	const float dx = 1;
	const float dt = 1.f / 60.f;
	const float alpha = 0;
	const float r = (alpha * dt) / (dx * dx);
	const float r2 = 1 - 2 * r;

	VALID(cudaMemcpyAsync(data, vec.data(), x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid(x_axis_bound / block.x, y_axis_bound / block.y);

	kernel_ftcs << <grid, block >> > (data, out_data, x_axis_bound, y_axis_bound, r, r2);

	VALID(cudaMemcpyAsync(vec.data(), out_data, x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyDeviceToHost));
}
