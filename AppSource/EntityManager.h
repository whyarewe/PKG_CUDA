#pragma once
#include "Config.h"
#include "IEntityManager.h"

namespace CoreUtils
{
	class EntityManager : public IEntityManager
	{
	private:
		Swarm swarm_;
		Swarm volatile_elements_;
		uint16_t radius_ { Config::Entity::default_entity_radius };

	public:
		EntityManager();
		~EntityManager();
		auto killLast() -> void override;
		auto killFirst() -> void override;
		auto decreaseRadius() -> void override;
		auto increaseRadius() -> void override;
		auto number() const->uint32_t override;
		auto getAll() const -> const Swarm& override;
		auto clearTemporaryElements() -> void override;
		auto getTemporary() const -> const Swarm& override;
		auto getCurrentRadius() const -> uint16_t override;
		auto spawn(const sf::Vector2i&, uint32_t, uint32_t) -> void override;
		auto spawnTemporary(const sf::Vector2i&, uint32_t, uint32_t) -> void override;
		
	};

}