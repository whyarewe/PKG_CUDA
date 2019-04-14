#include <cstdlib>

#include "CUDAPropagation.h"
#include "CUDAUtils.h"
#include "Config.h"

using namespace CUDAHelpers;

CUDAPropagation::CUDAPropagation(uint32_t x_axis_bound, uint32_t y_axis_bound)
{
	data = nullptr;
	out_data = nullptr;

	VALID(cudaMalloc(reinterpret_cast<void**>(&data), x_axis_bound * y_axis_bound * sizeof(float)));
	VALID(cudaMalloc(reinterpret_cast<void**>(&out_data), x_axis_bound * y_axis_bound * sizeof(float)));
}

CUDAPropagation::~CUDAPropagation()
{
	cudaFree(data);
	cudaFree(out_data);
}

auto CUDAPropagation::propagate(const ComputingData& data, const Device device, const Method method) -> void
{
	switch (method)
	{
	case Method::Laplace:
		laplace(data, device);
		break;
	case Method::FTCS:
		ftcs(data, device);
		break;
	case Method::FIS:
		dhe(data, device);
		break;
	default:
		std::cerr << "CUDA Propagation: Critical error, unknown method!" << std::endl;
		std::exit(0);
	}
}

auto CUDAPropagation::laplace(const ComputingData& data, const Device device) -> void
{
	switch (device)
	{
	case Device::CPU:
		laplace_cpu(data.board, data.x_axis_bound, data.y_axis_bound);
		break;
	case Device::GPU:
		laplace_gpu(data.board, data.x_axis_bound, data.y_axis_bound);
	}
}

auto CUDAPropagation::ftcs(const ComputingData& data, const Device device) -> void
{
	switch (device)
	{
	case Device::CPU:
		ftcs_cpu(data.board, data.x_axis_bound, data.y_axis_bound);
		break;
	case Device::GPU:
		ftcs_gpu(data.board, data.x_axis_bound, data.y_axis_bound);
	}
}

auto CUDAPropagation::dhe(const ComputingData& data, const Device device) -> void
{
	switch (device)
	{
	case Device::CPU:
		dhe_cpu(data.board, data.x_axis_bound, data.y_axis_bound);
		break;
	case Device::GPU:
		dhe_gpu(data.board, data.x_axis_bound, data.y_axis_bound);
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

__global__ void kernel_dhe(float* data, float* out_data, const int x_axis_bound, const int y_axis_bound, const float x_param, const float y_param)
{
	const uint16_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const uint16_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	const auto gid = idy * x_axis_bound + idx;

	if (idx > x_axis_bound || idy > y_axis_bound)
	{
		return;
	}

	float current_val = data[gid];

	float x_comp = x_param * (data[gid + x_axis_bound] - (2 * current_val) +
		data[gid - x_axis_bound]);

	float y_comp = y_param * (data[gid + 1] - (2 * current_val) +
		data[gid - 1]);

	out_data[gid] = current_val + x_comp + y_comp;
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

auto CUDAPropagation::laplace_gpu(std::vector<float>& vec, const uint32_t x_axis_bound,
	const uint32_t y_axis_bound) -> void
{
	VALID(cudaMemcpyAsync(data, vec.data(), x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid(x_axis_bound / block.x, y_axis_bound / block.y);

	kernel_laplace << <grid, block >> > (data, out_data, x_axis_bound, y_axis_bound);

	VALID(cudaMemcpyAsync(vec.data(), out_data, x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyDeviceToHost));
}

auto CUDAPropagation::ftcs_cpu(std::vector<float>& vec, const uint32_t x_axis_bound, const uint32_t y_axis_bound) -> void
{
	std::vector<float> out_vec(vec.capacity());

	const float r = (Config::FTCS_Config::alpha * Config::FTCS_Config::dt) / (Config::FTCS_Config::dx * Config::FTCS_Config::dx);
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

auto CUDAPropagation::ftcs_gpu(std::vector<float>& vec, const uint32_t x_axis_bound,
	const uint32_t y_axis_bound) -> void
{
	const float r = (Config::FTCS_Config::alpha * Config::FTCS_Config::dt) / (Config::FTCS_Config::dx * Config::FTCS_Config::dx);
	const float r2 = 1 - 2 * r;

	VALID(cudaMemcpyAsync(data, vec.data(), x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid(x_axis_bound / block.x, y_axis_bound / block.y);

	kernel_ftcs << <grid, block >> > (data, out_data, x_axis_bound, y_axis_bound, r, r2);

	VALID(cudaMemcpyAsync(vec.data(), out_data, x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyDeviceToHost));
}

auto CUDAPropagation::dhe_cpu(std::vector<float>& vec, const uint32_t x_axis_bound, const uint32_t y_axis_bound) -> void
{
	std::vector<float> out_vec(vec.capacity());

	float x_param = ((Config::DHE_Config::K * Config::DHE_Config::dt) /
		(Config::DHE_Config::sh * Config::DHE_Config::density * Config::DHE_Config::dx * Config::DHE_Config::dx));

	float y_param = ((Config::DHE_Config::K * Config::DHE_Config::dt) /
		(Config::DHE_Config::sh * Config::DHE_Config::density * Config::DHE_Config::dy * Config::DHE_Config::dy));

	for (auto i = 1u; i < y_axis_bound - 1; ++i)
	{
		for (auto j = 1u; j < x_axis_bound - 1; j++)
		{
			float current_val = vec[i * x_axis_bound + j];

			float x_comp = x_param * (vec[(i + 1) * x_axis_bound + j] - (2 * current_val) +
				vec[(i - 1) * x_axis_bound + j]);

			float y_comp = y_param * (vec[i * x_axis_bound + j + 1] - (2 * current_val) +
				vec[i * x_axis_bound + j - 1]);

			out_vec[i * x_axis_bound + j] = current_val + x_comp + y_comp;
		}
	}

	vec = out_vec;
}

auto CUDAPropagation::dhe_gpu(std::vector<float>& vec, const uint32_t x_axis_bound,
	const uint32_t y_axis_bound) -> void
{
	float x_param = ((Config::DHE_Config::K * Config::DHE_Config::dt) /
		(Config::DHE_Config::sh * Config::DHE_Config::density * Config::DHE_Config::dx * Config::DHE_Config::dx));

	float y_param = ((Config::DHE_Config::K * Config::DHE_Config::dt) /
		(Config::DHE_Config::sh * Config::DHE_Config::density * Config::DHE_Config::dy * Config::DHE_Config::dy));

	VALID(cudaMemcpyAsync(data, vec.data(), x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid(x_axis_bound / block.x, y_axis_bound / block.y);

	kernel_dhe << <grid, block >> > (data, out_data, x_axis_bound, y_axis_bound, x_param, y_param);

	VALID(cudaMemcpyAsync(vec.data(), out_data, x_axis_bound * y_axis_bound * sizeof(float), cudaMemcpyDeviceToHost));
}