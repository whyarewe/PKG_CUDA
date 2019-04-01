#include "Entity.h"
#include "EntityManager.h"

CoreUtils::EntityManager::EntityManager() = default;

CoreUtils::EntityManager::~EntityManager() = default;

auto CoreUtils::EntityManager::killLast() -> void
{
	if (!swarm_.empty()) { swarm_.pop_back(); }
}

auto CoreUtils::EntityManager::killFirst() -> void
{
	if (!swarm_.empty()) { swarm_.erase(swarm_.begin()); }
}

auto CoreUtils::EntityManager::getAll() const -> const Swarm&
{
	return swarm_;
}

auto CoreUtils::EntityManager::decreaseRadius() -> void
{
	if (radius_ > Config::Entity::minimal_entity_radius) { radius_--; }
}

auto CoreUtils::EntityManager::increaseRadius() -> void
{
	if (radius_ < Config::Entity::maximal_entity_radius) { radius_++; }
}

auto CoreUtils::EntityManager::number() const -> uint32_t
{
	return static_cast<uint32_t>(swarm_.size());
}

auto CoreUtils::EntityManager::getCurrentRadius() const -> uint16_t
{
	return radius_;
}

auto CoreUtils::EntityManager::clearTemporaryElements() -> void
{
	volatile_elements_.clear();
}

auto CoreUtils::EntityManager::getTemporary() const -> const Swarm&
{
	return volatile_elements_;
}

auto CoreUtils::EntityManager::spawn(const sf::Vector2i& position, const uint32_t x_axis_bound,
                                     const uint32_t y_axis_bound) -> void
{
	swarm_.push_back(Entity(position.x, position.y, radius_, x_axis_bound, y_axis_bound));
}

auto CoreUtils::EntityManager::spawnTemporary(const sf::Vector2i& position, const uint32_t x_axis_bound,
                                              const uint32_t y_axis_bound) -> void
{
	volatile_elements_.push_back(Entity(position.x, position.y, radius_, x_axis_bound, y_axis_bound));
}
