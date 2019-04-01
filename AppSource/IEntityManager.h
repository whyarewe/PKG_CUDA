#pragma once
#include <vector>
#include <cstdint>

#include <SFML/System/Vector2.hpp>

#include "Entity.h"

namespace CoreUtils
{
	class IEntityManager
	{
	public:
		using Swarm = std::vector<Entity>;

		IEntityManager() = default;
		virtual ~IEntityManager() = default;
		IEntityManager(const IEntityManager&) = delete;
		IEntityManager(const IEntityManager&&) = delete;
		IEntityManager& operator=(const IEntityManager&) = delete;
		IEntityManager& operator=(const IEntityManager&&) = delete;

		virtual auto killLast()->void = 0;
		virtual auto killFirst()-> void = 0;
		virtual auto decreaseRadius()-> void = 0;
		virtual auto increaseRadius() -> void = 0;
		virtual auto number() const->uint32_t = 0;
		virtual auto getAll() const ->const Swarm& = 0;
		virtual auto clearTemporaryElements() -> void = 0;
		virtual auto getCurrentRadius() const ->uint16_t = 0;
		virtual auto getTemporary() const -> const Swarm& = 0;
		virtual auto spawn(const sf::Vector2i&, uint32_t, uint32_t) -> void = 0;
		virtual auto spawnTemporary(const sf::Vector2i&, uint32_t, uint32_t) -> void = 0;
	};
}