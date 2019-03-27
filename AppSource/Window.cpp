#include "Window.h"

CoreUtils::Window::Window(WindowStyles style): window_style_(style)
{
	settings_ = sf::ContextSettings();
	settings_.antialiasingLevel = Config::StandardWindowSetting::anti_aliasing_level;
	window_ = std::make_unique<sf::RenderWindow>(
		sf::VideoMode(Config::StandardResolution::width, Config::StandardResolution::height), Config::project_name,
		static_cast<uint32_t>(style), settings_);
	window_->setFramerateLimit(Config::StandardWindowSetting::frame_rate_limit);
}

auto CoreUtils::Window::close() -> void
{
	if (window_->isOpen()) { window_->close(); }
}

auto CoreUtils::Window::getWidth() const -> uint32_t
{
	if (window_->isOpen()) { return window_->getSize().x; }
	return  0;
}

auto CoreUtils::Window::getHeight() const -> uint32_t
{
	if (window_->isOpen()) { return window_->getSize().y; }
	return  0;
}

auto CoreUtils::Window::getStyle() const -> WindowStyles
{
	return window_style_;
}

auto CoreUtils::Window::setStyle(const WindowStyles new_style) -> void
{
	if (window_->isOpen())
	{
		window_->close();
				
		if (WindowStyles::Resizable == new_style || WindowStyles::NonResizable == new_style)
		{
			window_->create(sf::VideoMode(Config::StandardResolution::width, Config::StandardResolution::height),
				Config::project_name, static_cast<uint32_t>(new_style), settings_);
		}
		else if (WindowStyles::FullScreen == new_style)
		{
			window_->create(sf::VideoMode(Config::FullHDResolution::width, Config::FullHDResolution::height),
				Config::project_name, static_cast<uint32_t>(new_style), settings_);
		}
	}
}

auto CoreUtils::Window::getMousePosition() const -> sf::Vector2i
{
	return sf::Mouse::getPosition(*window_);
}

auto CoreUtils::Window::clear() -> void
{
	if (window_->isOpen()) { window_->clear(); }
}

auto CoreUtils::Window::draw(sf::Drawable* object) -> void
{
	if (window_->isOpen()) { window_->draw(*object); }
}

auto CoreUtils::Window::pollEvent(sf::Event& event) -> bool
{
	if (window_->isOpen()) { return window_->pollEvent(event); }
	return  false;
}

auto CoreUtils::Window::display() -> void
{
	if (window_->isOpen()) { window_->display(); }
}

auto CoreUtils::Window::isOpen() -> bool
{
	return window_->isOpen();
}