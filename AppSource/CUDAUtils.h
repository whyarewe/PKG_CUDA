#pragma once

#include <iostream>
#include "cuda_runtime.h"

namespace CUDAHelpers
{
	template <typename T>
	void VALID(T expression)
	{
		if (cudaSuccess != expression)
		{
			std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(expression) << std::endl;
			std::exit(1);
		}
	}
}
