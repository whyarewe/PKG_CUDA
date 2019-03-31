#pragma once
#include <cstdint>
#include <atomic>

#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/RenderWindow.hpp>

#include "Config.h"
#include "IGUI.h"

namespace CoreUtils
{
	class GUI : public IGUI
	{
	private:
		std::atomic<uint16_t> radius_{1};
		std::atomic<uint32_t> heaters_count_{0};
		std::atomic<bool> show_controls_{false};

		std::unique_ptr<sf::Text> radius_text_;
		std::unique_ptr<sf::Text> heaters_count_text_;
		std::unique_ptr<sf::Text> info_text_;
		std::unique_ptr<sf::Text> key_bindings_text_;

	public:
		GUI(const sf::RenderWindow& window, const sf::Font&);
		auto update() -> void override;
		auto setRadius(uint16_t) -> void override;
		auto toggleShowControls() -> void override;
		auto display(sf::RenderWindow&) -> void override;
		auto setHeatersCount(size_t) -> void override;
		auto setFontConfiguration(const sf::Font&) const -> void override;
	};
}