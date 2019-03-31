#pragma once

#include <vector>

#include "Entity.h"

namespace CUDAHelpers
{
	struct ComputingData
	{
		std::vector<float>& board;
		uint32_t x_axis_bound;
		uint32_t y_axis_bound;
		uint16_t entity_radius;
		Entity::EntityContainer swarm;
	};

	class CUDAPropagation
	{
	public:
		enum class Device { CPU, GPU };

		static void laplace(ComputingData data, Device device);

	private:
		static auto laplace_cpu(std::vector<float>& vec, uint32_t x_axis_bound, uint32_t y_axis_bound,
		                        Entity::EntityContainer swarm) -> void;

		static auto laplace_gpu(std::vector<float>& vec, uint32_t x_axis_bound, uint32_t y_axis_bound) -> void;
	};
}
