#include "CUDASystemInformation.h"

using namespace CUDAHelpers;

CUDASystemInformation::CUDASystemInformation()
{
	cudaGetDeviceCount(&this->device_count_);
	for (uint8_t deviceIndex = 0; deviceIndex < this->device_count_; ++deviceIndex)
	{
		cudaDeviceProp properties{};
		cudaGetDeviceProperties(&properties, deviceIndex);
		this->devices_.emplace(std::string(properties.name), properties);
	}
}

auto CUDASystemInformation::getNumberOfDevices() const -> int
{
	return this->device_count_;
}

auto CUDASystemInformation::getDeviceProperties(const std::string& name) const -> cudaDeviceProp
{
	return this->devices_.at(name);
}

auto CUDASystemInformation::displaySystemDevicesProperties() const -> void
{
	auto result = getDevicesPropertiesAsFormattedText();
	std::cout << result.str();
}

auto CUDASystemInformation::getDevicesPropertiesAsFormattedText() const -> std::stringstream
{
	std::stringstream result;

	for each (const auto &device in devices_)
	{
		result << device.first << std::endl;
		result << "|  Compute capability: " << device.second.major << "." << device.second.minor << std::endl;

		result << "|  Max thread per blocks: " << device.second.maxThreadsPerBlock << std::endl;
		result << "|  Max grid dimensions: ("
			<< device.second.maxGridSize[0] << " x "
			<< device.second.maxGridSize[1] << " x "
			<< device.second.maxGridSize[2] << ")" << std::endl;

		result << "|  Max block dimensions: ("
			<< device.second.maxThreadsDim[0] << " x "
			<< device.second.maxThreadsDim[1] << " x "
			<< device.second.maxThreadsDim[2] << ")" << std::endl;
		result << "-" << std::endl;
	}

	return result;
}
