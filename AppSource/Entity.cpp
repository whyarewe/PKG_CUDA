#include "Entity.h"

Coordinates Entity::getCoordinates() const
{
	return coordinates_;
}

uint32_t Entity::getRadius() const
{
	return radius_;
}

void Entity::setCoordinates(uint32_t x, uint32_t y)
{
	coordinates_.setX(x);
	coordinates_.setY(y);
}

void Entity::setCoordinates(Coordinates coordinates)
{
	coordinates_ = coordinates;
}

void Entity::setRadius(uint32_t radius)
{
	radius_ = radius;
}

Entity::Entity(Coordinates coordinates, uint16_t radius) :
	coordinates_(coordinates), radius_(radius) 
{

}

Entity::Entity(uint32_t x, uint32_t y, uint16_t radius) :
	radius_(radius)
{
	coordinates_ = Coordinates(x, y);	
}
