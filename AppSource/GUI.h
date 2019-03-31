#pragma once
#include "Config.h"
#include "IGUI.h"

#include <cstdint>
#include <atomic>

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Font.hpp>

namespace CoreUtils
{
	class GUI : public IGUI
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
		auto setFontConfiguration(const sf::Font&) const -> void override;
		auto setRadius(uint16_t) -> void override;
		auto setHeatersCount(uint16_t) -> void override;
		auto toggleShowControls() -> void override;
		auto update() -> void;
		auto display(sf::RenderWindow&) -> void override;
	};
}