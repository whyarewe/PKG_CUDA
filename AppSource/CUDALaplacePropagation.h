#pragma once

#include <cstdio>
#include <cstdint>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CUDALaplacePropagation
{
	void propagate(std::vector<float> &vec, int xAxisBound, int yAxisBound, int xHeaterPos, int yHeaterPos);
};
