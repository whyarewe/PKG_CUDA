#pragma once
#include <cstdint>

#include "Config.h"
#include "SFML/Window.hpp"
#include <SFML/Graphics/Drawable.hpp>
#include "CUDALaplacePropagation.h"
#include <SFML/Graphics/Image.hpp>

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
		IWindow(const IWindow&) = delete;
		IWindow(const IWindow&&) = delete;
		IWindow &operator=(const IWindow&) = delete;
		IWindow &operator=(const IWindow&&) = delete;

		virtual auto close()-> void = 0;
		virtual auto clear()->void = 0;
		virtual auto display()-> void = 0;
		virtual auto pollEvent(sf::Event&)-> bool = 0;
		virtual auto draw(sf::Drawable*)->void = 0;
		
		virtual auto isOpen()->bool = 0;
		virtual auto getWidth() const->uint32_t = 0;
		virtual auto getHeight() const->uint32_t = 0;
		virtual auto getStyle() const->WindowStyles = 0;
		virtual auto setStyle(WindowStyles) -> void = 0;

		virtual auto getMousePosition() const->sf::Vector2i = 0;
		virtual auto calculateView(const CUDAHelpers::ComputingData&) -> void = 0;
		virtual auto constructImageFromVector(const CUDAHelpers::ComputingData&) const->sf::Image = 0;
		virtual auto setActive(bool) const -> void = 0;

		virtual ~IWindow() = default;
	};
}
