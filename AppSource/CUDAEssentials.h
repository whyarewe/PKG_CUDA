#pragma once

#include <cstdio>
#include <cstdint>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CUDAHelpers {
	typedef struct {
		uint8_t R;
		uint8_t G;
		uint8_t B;
	} RGB;

	class CUDAEssentials
	{
	public:
		static void writeP6_PPM(const char* filename, RGB* pixelData, int width, int height);
		static void generateImage(RGB *image, int width, int height);
	};
}
