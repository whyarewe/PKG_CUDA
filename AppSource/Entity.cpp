#include "Entity.h"

auto Entity::getDimensions() const -> EntityDimensions
{
	return dimensions_;
}

Entity::Entity(const uint32_t x, const uint32_t y, const uint32_t radius, const uint32_t x_axis_bound,
               const uint32_t y_axis_bound) :
	dimensions_(x, y, radius, x_axis_bound, y_axis_bound)
{
}

Entity::~Entity() = default;
