#pragma once
#include <atomic>
#include <cstdint>

#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/RenderWindow.hpp>

#include "Config.h"

namespace CoreUtils
{
	class IGUI
	{
	public:
		IGUI() = default;
		virtual ~IGUI() = default;
		IGUI(const IGUI&) = delete;
		IGUI(const IGUI&&) = delete;
		IGUI& operator=(const IGUI&) = delete;
		IGUI& operator=(const IGUI&&) = delete;

		virtual auto update() -> void = 0;
		virtual auto setRadius(uint16_t) -> void = 0;
		virtual auto toggleShowControls() -> void = 0;
		virtual auto setHeatersCount(size_t) -> void = 0;
		virtual auto display(sf::RenderWindow&) -> void = 0;
		virtual auto setFontConfiguration(const sf::Font&) const -> void = 0;
		virtual auto getDeviceName() -> std::string = 0;
		virtual auto getMethodName()->std::string = 0;
	};
}
