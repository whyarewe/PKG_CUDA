#pragma once
#include "IEntityManager.h"

namespace CoreUtils
{
	class ILevelManager
	{
	public:
		using Level = std::vector<float>;

		ILevelManager() = default;
		virtual ~ILevelManager() = default;
		ILevelManager(const ILevelManager&) = delete;
		ILevelManager(const ILevelManager&&) = delete;
		ILevelManager& operator=(const ILevelManager&) = delete;
		ILevelManager& operator=(const ILevelManager&&) = delete;

		auto virtual getLevel() -> Level& = 0;
		auto virtual viewLevel() const -> const Level& = 0;
		auto virtual update(IEntityManager&) -> void = 0;
		auto virtual getXAxisLength() const -> uint32_t = 0;
		auto virtual getYAxisLength() const -> uint32_t = 0;
	};
}
