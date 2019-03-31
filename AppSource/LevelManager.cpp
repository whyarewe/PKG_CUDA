#include "LevelManager.h"

auto CoreUtils::LevelManager::setHeatLevel(const IEntityManager::Swarm& swarm, const uint32_t x_axis_bound) -> void
{
	for (const auto& entity : swarm)
	{
		for (auto i = entity.getDimensions().getTopBorder(); i < entity.getDimensions().getBottomBorder(); ++i)
		{
			for (auto j = entity.getDimensions().getLeftBorder(); j < entity.getDimensions().getRightBorder(); ++j)
			{
				level_[i * x_axis_bound + j] = 255.f;
			}
		}
	}
}

CoreUtils::LevelManager::LevelManager(const uint32_t x_axis_size, const uint32_t y_axis_size) :
	x_axis_length_(x_axis_size), y_axis_length_(y_axis_size)
{
	level_.reserve(x_axis_size*y_axis_size);
}

CoreUtils::LevelManager::~LevelManager() = default;

auto CoreUtils::LevelManager::getLevel() -> Level&
{
	return level_;
}

auto CoreUtils::LevelManager::update(IEntityManager& entity_manager) -> void
{
	setHeatLevel(entity_manager.getTemporary(), x_axis_length_);
	entity_manager.clearTemporaryElements();
	setHeatLevel(entity_manager.getAll(), x_axis_length_);
}

auto CoreUtils::LevelManager::getXAxisLength() const -> uint32_t
{
	return x_axis_length_;
}

auto CoreUtils::LevelManager::getYAxisLength() const -> uint32_t
{
	return y_axis_length_;
}
