#pragma once

#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CUDAHelpers
{
	class CUDASystemInformation
	{
	public:
		using SystemDevices = std::map<std::string, cudaDeviceProp>;

		int getNumberOfDevices() const;
		void displaySystemDevicesProperites() const;
		cudaDeviceProp getDeviceProperties(std::string) const;
		CUDASystemInformation();

	private:
		int _deviceCount = 0;
		SystemDevices _devices;
		std::stringstream getDevicesPropertiesAsFormattedText() const;
	};
};

