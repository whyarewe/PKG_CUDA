﻿#pragma once
#include <cstdint>

#include "Config.h"
#include "GUI.h"
#include "SFML/Window.hpp"
#include <SFML/Graphics/Drawable.hpp>
#include "CUDALaplacePropagation.h"
#include <SFML/Graphics/Image.hpp>

namespace sf
{
	class Font;
}

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
		virtual auto generateView(const CUDAHelpers::ComputingData&) -> void = 0;
		virtual auto constructImageFromVector(sf::Image& background_image,
		                                      const CUDAHelpers::ComputingData& data) const -> sf::Image = 0;

	};
}
