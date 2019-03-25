#pragma once

#include <cstdint>
#include <vector>

class Coordinates {	
private:
	uint32_t x_;
	uint32_t y_;
public: 
	Coordinates() = default;
	Coordinates(uint32_t x, uint32_t y) : x_(x), y_(y) {};

	uint32_t getX() const { return x_; }
	uint32_t getY() const { return y_; }
	void setX(uint32_t x) { x_ = x; }
	void setY(uint32_t y) { y_ = y; }

	~Coordinates() = default;
};

class Entity
{
private:
	Coordinates coordinates_;
	uint32_t radius_;
public:
	using EntityContainer = std::vector<Entity>;

	Coordinates getCoordinates() const;
	uint32_t getRadius() const;
	void setCoordinates(uint32_t x, uint32_t y);
	void setCoordinates(Coordinates coordinates);
	void setRadius(uint32_t radius);

	Entity() = delete;
	explicit Entity(Coordinates coordinates, uint16_t radius = 1);
	Entity(uint32_t x, uint32_t y, uint16_t radius = 1);
	~Entity() = default;
};

