#pragma once
#include <map>
#include <string>
#include <sstream>
#include <iostream>

#include "cuda_runtime.h"

namespace CUDAHelpers
{
	class CUDASystemInformation
	{
	public:
		using SystemDevices = std::map<std::string, cudaDeviceProp>;

		auto getNumberOfDevices() const -> int;
		auto displaySystemDevicesProperties() const -> void;
		auto getDeviceProperties(const std::string&) const -> cudaDeviceProp;
		CUDASystemInformation();

	private:
		int device_count_ = 0;
		SystemDevices devices_;
		auto getDevicesPropertiesAsFormattedText() const -> std::stringstream;
	};
};
