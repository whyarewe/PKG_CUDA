#pragma once
#include "IWindow.h"

#include <SFML/Graphics/RenderWindow.hpp>

namespace CoreUtils
{
	class Window : public IWindow
	{
	private:
		std::unique_ptr<sf::RenderWindow> window_;
		WindowStyles window_style_;
		sf::ContextSettings settings_;
	public:
		explicit Window(WindowStyles);		
		auto close() -> void override;
		auto clear() -> void override;
		auto isOpen() -> bool override;
		auto display() -> void override;
		auto draw(sf::Drawable*)-> void override;
		auto pollEvent(sf::Event&) -> bool override;
		auto getWidth() const -> uint32_t override;
		auto getHeight() const -> uint32_t override;
		auto getStyle() const->WindowStyles override;
		auto setStyle(WindowStyles) -> void override;
		auto getMousePosition() const->sf::Vector2i override;

		// ~Window() = default;
	};
}