#pragma once
#include <cstdint>

#include <SFML/Window.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics/Drawable.hpp>

#include "GUI.h"
#include "Color.h"
#include "Config.h"
#include "CUDAPropagation.h"

namespace CoreUtils
{
	enum class WindowStyles
	{
		Resizable = sf::Style::Resize | sf::Style::Close,
		NonResizable = sf::Style::Close,
		FullScreen = sf::Style::Fullscreen
	};

	class IWindow
	{
	public:
		IWindow() = default;
		virtual ~IWindow() = default;
		IWindow(const IWindow&) = delete;
		IWindow(const IWindow&&) = delete;
		IWindow& operator=(const IWindow&) = delete;
		IWindow& operator=(const IWindow&&) = delete;

		virtual auto clear() -> void = 0;
		virtual auto close() -> void = 0;
		virtual auto isOpen() -> bool = 0;
		virtual auto display() -> void = 0;
		virtual auto isWithinWindow(const sf::Vector2i&) -> bool = 0;
		virtual auto reloadWindow() -> void = 0;
		virtual auto updateInterface() -> void = 0;
		virtual auto toggleControls() -> void = 0;
		virtual auto draw(sf::Drawable*) -> void = 0;
		virtual auto getWidth() const -> uint32_t = 0;
		virtual auto getHeight() const -> uint32_t = 0;
		virtual auto pollEvent(sf::Event&) -> bool = 0;
		virtual auto setActive(bool) const -> void = 0;
		virtual auto setStyle(WindowStyles) -> void = 0;
		virtual auto getStyle() const -> WindowStyles = 0;
		virtual auto getMousePosition() const -> sf::Vector2i = 0;
		virtual auto generateView(const ILevelManager&, const IEntityManager&) -> void = 0;
		virtual auto constructImageFromVector(std::vector<Color>&, const ILevelManager&) const -> void = 0;
	};
}
