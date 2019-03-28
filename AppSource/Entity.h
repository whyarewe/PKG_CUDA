#pragma once

#include <cstdint>
#include <vector>

class EntityDimensions
{
private:
	uint32_t x_;
	uint32_t y_;
	uint32_t top_border_;
	uint32_t left_border_;
	uint32_t right_border_;
	uint32_t bottom_border_;
	uint32_t radius_;

public:
	EntityDimensions() = default;

	EntityDimensions(const uint32_t x, const uint32_t y) : x_(x), y_(y)
	{
		radius_ = 1;
		top_border_ = y - 1;
		left_border_ = x - 1;
		right_border_ = x + 1;
		bottom_border_ = y + 1;
	};

	EntityDimensions(const uint32_t x, const uint32_t y, const uint32_t radius, const uint32_t x_axis_bound,
	                 const uint32_t y_axis_bound) : x_(x), y_(y), radius_(radius)
	{
		top_border_ = y - radius_;
		left_border_ = x - radius_;
		right_border_ = x + radius_;
		bottom_border_ = y + radius_;
		top_border_ = top_border_ <= 1 ? 1 : top_border_;
		left_border_ = left_border_ <= 1 ? 1 : left_border_;
		right_border_ = right_border_ >= x_axis_bound ? right_border_ - 1 : right_border_;
		bottom_border_ = bottom_border_ >= y_axis_bound ? bottom_border_ - 1 : bottom_border_;
	};

	auto getX() const -> uint32_t { return x_; }
	auto getY() const -> uint32_t { return y_; }
	auto getTopBorder() const -> uint32_t { return top_border_; }
	auto getLeftBorder() const -> uint32_t { return left_border_; }
	auto getRightBorder() const -> uint32_t { return right_border_; }
	auto getBottomBorder() const -> uint32_t { return bottom_border_; }

	~EntityDimensions() = default;
};

class Entity
{
private:
	EntityDimensions dimensions_;
public:
	using EntityContainer = std::vector<Entity>;
	auto getDimensions() const -> EntityDimensions;

	Entity() = delete;
	Entity(uint32_t x, uint32_t y, uint32_t radius, uint32_t x_axis_bound,
	       uint32_t y_axis_bound);
	~Entity();
};
