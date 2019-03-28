#pragma once
#include "Config.h"

#include <cstdint>
#include <atomic>

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Font.hpp>

namespace CoreUtils
{
	class GUI
	{
	private:
		std::atomic<uint16_t> radius_{1};
		std::atomic<uint16_t> heaters_count_{0};
		std::atomic<bool> show_controls_{false};

		std::unique_ptr<sf::Text> radius_text_;
		std::unique_ptr<sf::Text> heaters_count_text_;
		std::unique_ptr<sf::Text> info_text_;
		std::unique_ptr<sf::Text> key_bindings_text_;
	public:

		GUI(sf::RenderWindow& window, const sf::Font&);
		auto setFontConfiguration(const sf::Font&) const -> void;
		auto setRadius(uint16_t) -> void;
		auto setHeatersCount(uint16_t) -> void;
		auto setShowControls(bool) -> void;
		auto update() -> void;
		auto display(sf::RenderWindow&) -> void;
	};
}