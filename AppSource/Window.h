#pragma once
#include "IWindow.h"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <thread>
#include <future>

namespace CoreUtils
{
	class Window : public IWindow
	{
	private:
		std::thread view_;
		WindowStyles window_style_;
		sf::ContextSettings settings_;		
		std::unique_ptr<sf::RenderWindow> window_;
		std::atomic<bool> running_view_{ false };
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

		auto calculateView(const CUDAHelpers::ComputingData&) -> void override;
		auto constructImageFromVector(const CUDAHelpers::ComputingData&) const -> sf::Image override;
		auto setActive(bool) const -> void override;

		~Window();	
	};
}
