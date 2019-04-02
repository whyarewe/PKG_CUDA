#pragma once
#include <thread>
#include <future>

#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/RenderWindow.hpp>

#include "IWindow.h"
#include "IGUI.h"

namespace CoreUtils
{
	class Window : public IWindow
	{
	private:
		std::thread view_;

		WindowStyles window_style_;
		std::unique_ptr<IGUI> gui_;
		sf::ContextSettings settings_;
		std::unique_ptr<sf::RenderWindow> window_;

		std::atomic<bool> running_view_{false};
		std::atomic<bool> needs_reload_{false};
		std::atomic<bool> update_interface_{false};

	public:
		explicit Window(WindowStyles, const sf::Font&);
		auto close() -> void override;
		auto clear() -> void override;
		auto isOpen() -> bool override;
		auto display() -> void override;
		auto reloadWindow() -> void override;
		auto updateInterface() -> void override;
		auto toggleControls() -> void override;
		auto draw(sf::Drawable*) -> void override;
		auto getWidth() const -> uint32_t override;
		auto pollEvent(sf::Event&) -> bool override;
		auto getHeight() const -> uint32_t override;
		auto setActive(bool) const -> void override;
		auto setStyle(WindowStyles) -> void override;
		auto getStyle() const -> WindowStyles override;
		auto getMousePosition() const -> sf::Vector2i override;
		auto isWithinWindow(const sf::Vector2i&) -> bool override;
		auto generateView(const ILevelManager&, const IEntityManager&) -> void override;
		auto constructImageFromVector(sf::Image& background_image, const ILevelManager&) const -> sf::Image override;

		~Window();
	};
}
