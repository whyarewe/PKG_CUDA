#pragma once
#include "ILevelManager.h"

namespace CoreUtils
{
	class LevelManager :
		public ILevelManager
	{
	private:
		Level level_;
		uint32_t x_axis_length_;
		uint32_t y_axis_length_;
		auto setHeatLevel(const IEntityManager::Swarm&, uint32_t) -> void;
	public:
		LevelManager() = delete;
		LevelManager(uint32_t x_axis_size, uint32_t y_axis_size);
		~LevelManager();
		auto getLevel() -> Level& override;
		auto update(IEntityManager&) -> void override;
		auto viewLevel() const -> const Level& override;
		auto getXAxisLength() const -> uint32_t override;
		auto getYAxisLength() const -> uint32_t override;
	};
}
