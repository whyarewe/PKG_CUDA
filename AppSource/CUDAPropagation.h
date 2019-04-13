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

		CUDAPropagation(uint32_t, uint32_t);

		void propagate(const ComputingData&, const Device device, const Method method);

	private:
		float* data;
		float* out_data;

		auto laplace(const ComputingData& data, const Device device) -> void;
		auto ftcs(const ComputingData& data, const Device device) -> void;
		auto fis(const ComputingData& data, const Device device) -> void;

		auto laplace_cpu(std::vector<float>&, const uint32_t, const uint32_t) -> void;
		auto laplace_gpu(std::vector<float>&, const uint32_t, const uint32_t) -> void;

		auto ftcs_cpu(std::vector<float>&, const uint32_t, const uint32_t) -> void;
		auto ftcs_gpu(std::vector<float>&, const uint32_t, const uint32_t) -> void;

		auto fis_cpu(std::vector<float>&, const uint32_t, const uint32_t) -> void;
		auto fis_gpu(std::vector<float>&, const uint32_t, const uint32_t) -> void;
	};
}
