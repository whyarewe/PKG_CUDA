#pragma once

#include <cstdio>
#include <cstdint>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Entity.h"

struct ComputingData {
	std::vector<float>& board;
	int xAxisBound;
	int yAxisBound;
	Entity::EntityContainer swarm;
};

class CUDAPropagation
{
public:
	enum class Device {CPU, GPU};
	static void laplace(ComputingData data, Device device);

private:
	static void laplace_cpu(std::vector<float> &vec, int xAxisBound, int yAxisBound, Entity::EntityContainer swarm);
	static void laplace_gpu(std::vector<float> &vec, int xAxisBound, int yAxisBound, int xHeaterPos, int yHeaterPos);
};
