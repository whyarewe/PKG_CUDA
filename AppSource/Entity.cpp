#include "Entity.h"

auto Entity::getCoordinates() const -> Coordinates
{
	return coordinates_;
}

auto Entity::getRadius() const -> uint32_t
{
	return radius_;
}

auto Entity::setCoordinates(const uint32_t x, const uint32_t y) -> void
{
	coordinates_.setX(x);
	coordinates_.setY(y);
}

auto Entity::setCoordinates(const Coordinates coordinates) -> void
{
	coordinates_ = coordinates;
}

auto Entity::setRadius(const uint32_t radius) -> void
{
	radius_ = radius;
}

Entity::Entity(const Coordinates coordinates, const uint16_t radius) :
	coordinates_(coordinates), radius_(radius)
{

}

Entity::Entity(const uint32_t x, const uint32_t y, const uint16_t radius) :
	coordinates_(x, y), radius_(radius)
{
}
