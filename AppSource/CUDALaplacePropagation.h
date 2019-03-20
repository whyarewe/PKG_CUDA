#pragma once

#include <cstdio>
#include <cstdint>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CUDALaplacePropagation
{
public:
	static void propagate(std::vector<float> &vec, int xAxisBound, int yAxisBound, int xHeaterPos, int yHeaterPos);
};
