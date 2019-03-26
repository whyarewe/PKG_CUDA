#pragma once

#include <cstdint>
#include <vector>

class Coordinates {	
private:
	uint32_t x_;
	uint32_t y_;
public: 
	Coordinates() = default;
	Coordinates(const uint32_t x, const uint32_t y) : x_(x), y_(y) {};

	auto getX() const -> uint32_t { return x_; }
	auto getY() const -> uint32_t { return y_; }
	auto setX(const uint32_t x) -> void { x_ = x; }
	auto setY(const uint32_t y) -> void { y_ = y; }

	~Coordinates() = default;
};

class Entity
{
private:
	Coordinates coordinates_;
	uint32_t radius_;
public:
	using EntityContainer = std::vector<Entity>;

	auto getCoordinates() const -> Coordinates;
	auto getRadius() const -> uint32_t;
	auto setCoordinates(uint32_t x, uint32_t y) -> void;
	auto setCoordinates(Coordinates coordinates) -> void;
	auto setRadius(uint32_t radius) -> void;

	Entity() = delete;
	explicit Entity(Coordinates coordinates, uint16_t radius = 1);
	Entity(uint32_t x, uint32_t y, uint16_t radius = 1);
	~Entity() = default;
};

