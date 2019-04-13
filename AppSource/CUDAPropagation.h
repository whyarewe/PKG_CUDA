#pragma once
#include <vector>

#include "Entity.h"
#include "ILevelManager.h"

namespace CUDAHelpers
{
	struct ComputingData
	{
		std::vector<float>& board;
		uint32_t x_axis_bound;
		uint32_t y_axis_bound;
		uint16_t entity_radius;
		CoreUtils::Entity::EntityContainer swarm;
	};

	class CUDAPropagation
	{
	public:
		enum class Device { CPU, GPU };

		enum class Method { Laplace, FTCS, FIS };

		static void propagate(float* in, float* out, const ComputingData&, const Device device, const Method method);

	private:
		static auto laplace(float* in, float* out, const ComputingData& data, const Device device) -> void;
		static auto ftcs(float* in, float* out, const ComputingData& data, const Device device) -> void;
		static auto fis(float* in, float* out, const ComputingData& data, const Device device) -> void;

		static auto laplace_cpu(std::vector<float>&, const uint32_t, const uint32_t) -> void;
		static auto laplace_gpu(float*, float*, const uint32_t, const uint32_t, std::vector<float>&) -> void;

		static auto ftcs_cpu(std::vector<float>&, const uint32_t, const uint32_t) -> void;
		static auto ftcs_gpu(float*, float*, const uint32_t, const uint32_t, std::vector<float>&) -> void;

		static auto fis_cpu(std::vector<float>&, const uint32_t, const uint32_t) -> void;
		static auto fis_gpu(float*, float*, const uint32_t, const uint32_t, std::vector<float>&) -> void;
	};
}
