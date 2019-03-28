#pragma once

#include <vector>

#include "Entity.h"

namespace CUDAHelpers
{
	struct ComputingData
	{
		std::vector<float>& board;
		int x_axis_bound;
		int y_axis_bound;
		int entity_radius;
		bool show_controls;
		Entity::EntityContainer swarm;
	};

	class CUDAPropagation
	{
	public:
		enum class Device { CPU, GPU };

		static void laplace(ComputingData data, Device device);

	private:
		static auto laplace_cpu(std::vector<float>& vec, int x_axis_bound, int y_axis_bound,
		                        Entity::EntityContainer swarm) -> void;

		static auto laplace_gpu(std::vector<float>& vec, int x_axis_bound, int y_axis_bound) -> void;
	};
}
