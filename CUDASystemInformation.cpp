#include "CUDASystemInformation.h"

using namespace CUDAHelpers;

CUDASystemInformation::CUDASystemInformation()
{
	cudaGetDeviceCount(&this->_deviceCount);
	for (uint8_t deviceIndex = 0; deviceIndex < this->_deviceCount; ++deviceIndex)
	{
		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, deviceIndex);
		this->_devices.emplace(std::string(properties.name), properties);
	}
}

int CUDASystemInformation::getNumberOfDevices() const
{
	return this->_deviceCount;
}

cudaDeviceProp CUDASystemInformation::getDeviceProperties(std::string name) const
{
	return this->_devices.at(name);
}

void CUDASystemInformation::displaySystemDevicesProperites() const
{
	std::stringstream result = getDevicesPropertiesAsFormattedText();
	std::cout << result.str();
}

std::stringstream CUDASystemInformation::getDevicesPropertiesAsFormattedText() const
{
	std::stringstream result;

	for each (const auto &device in _devices)
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