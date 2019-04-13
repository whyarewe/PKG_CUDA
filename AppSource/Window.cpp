#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/Shader.hpp>

#include <iostream>

#include "Window.h"

namespace
{
	double R(const float x)
	{
		return 255.0605 + (0.02909945 - 255.0605) / std::pow((1 + std::pow((x / 68872.05), 2.133224)), 13205500);
	}

	auto G(const float x) -> double
	{
		return 10 + 6.109578 * x * 1.2 - 0.2057529 * x * x + 0.002335796 * x * x * x - 0.00001016682 * x * x * x * x +
			1.514604e-8 * x * x * x * x * x;
	}

	auto B(const float x) -> double
	{
		return 10 + 3.000859 * x - 0.09096409 * x * x + 0.0006806883 * x * x * x * 2 - 0.000001399089 * x * x * x * x;
	}
}

CoreUtils::Window::Window(WindowStyles style, const sf::Font& font) :
	window_style_(style)
{
	settings_ = sf::ContextSettings();
	settings_.antialiasingLevel = Config::StandardWindowSetting::anti_aliasing_level;
	window_ = std::make_unique<sf::RenderWindow>(
		sf::VideoMode(Config::StandardResolution::width, Config::StandardResolution::height), Config::project_name,
		static_cast<uint32_t>(style), settings_);
	window_->setFramerateLimit(Config::StandardWindowSetting::frame_rate_limit);

	system_font_ = font;
	gui_ = std::make_unique<GUI>(*window_, font);


	for (auto f =  0.f; f < 251; f += 0.1f) {
		r_tab_.push_back(static_cast<sf::Uint8>(R(f)));
		g_tab_.push_back(static_cast<sf::Uint8>(G(f)));
		b_tab_.push_back(static_cast<sf::Uint8>(B(f)));
	}
}

CoreUtils::Window::~Window()
{
	view_.join();
}

auto CoreUtils::Window::close() -> void
{
	if (window_->isOpen()) { window_->close(); }
}

auto CoreUtils::Window::getWidth() const -> uint32_t
{
	if (window_->isOpen()) { return window_->getSize().x; }
	return 0;
}

auto CoreUtils::Window::getHeight() const -> uint32_t
{
	if (window_->isOpen()) { return window_->getSize().y; }
	return 0;
}

auto CoreUtils::Window::getStyle() const -> WindowStyles
{
	return window_style_;
}

auto CoreUtils::Window::isWithinWindow(const sf::Vector2i& coordinates) -> bool
{
	return (coordinates.x > 1 && coordinates.x < getWidth() && coordinates.y > 1 && coordinates.y < getHeight());
}

auto CoreUtils::Window::setStyle(const WindowStyles new_style) -> void
{
	if (window_->isOpen())
	{
		needs_reload_ = true;
		using namespace std::chrono_literals;
		std::this_thread::sleep_for(50ms);
		window_->setActive(true);
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

		window_style_ = new_style;
		settings_ = sf::ContextSettings();
		settings_.antialiasingLevel = Config::StandardWindowSetting::anti_aliasing_level;
		window_->setFramerateLimit(Config::StandardWindowSetting::frame_rate_limit);
		needs_reload_ = false;
		gui_.reset(new GUI(*window_, system_font_));
	}
}

auto CoreUtils::Window::getMousePosition() const -> sf::Vector2i
{
	return sf::Mouse::getPosition(*window_);
}

auto CoreUtils::Window::generateView(const ILevelManager& level_manager, const IEntityManager& entity_manager) -> void
{
	if (running_view_) { return; }

	if (view_.joinable())
	{
		view_.join();
	}
	view_ = std::thread([&]()
	{
		running_view_ = true;

		std::vector<Color> background_values(level_manager.getXAxisLength() * level_manager.getYAxisLength(),
		                                     {0, 0, 0, 0});
		sf::Texture background_texture;
		background_texture.create(level_manager.getXAxisLength(), level_manager.getYAxisLength());
		background_texture.setSmooth(true);
		sf::Sprite background;
		background.setTexture(background_texture, false);

		while (isOpen() && !needs_reload_)
		{
			constructImageFromVector(background_values, level_manager);
			background_texture.update(reinterpret_cast<const sf::Uint8*>(background_values.data()));

			clear();
			sf::Shader shader;
			sf::RenderStates states;
			states.shader = &shader;
			window_->draw(background, states);

			if (update_interface_)
			{
				gui_->setRadius(entity_manager.getCurrentRadius());
				gui_->setHeatersCount(entity_manager.number());
				gui_->update();

				update_interface_ = false;
			}

			gui_->display(*window_);
			display();
		}
		window_->setActive(false);
		running_view_ = false;		
	});


}

auto CoreUtils::Window::constructImageFromVector(std::vector<Color>& texture_data,
                                                 const ILevelManager& level_manager) const -> void
{
	for (auto i = 1u; i < level_manager.getYAxisLength() - 1; ++i)
	{
		for (auto j = 1u; j < level_manager.getXAxisLength() - 1; ++j)
		{
			const auto current_index = i * level_manager.getXAxisLength() + j;
			const auto point_value = level_manager.viewLevel()[current_index];

			if (point_value > 125.f)
			{
				texture_data.at(current_index) = {255, 255, 255, 255};
			}
			else
			{
				texture_data.at(current_index) = {
					r_tab_[static_cast<int>(point_value * 10)],
					g_tab_[static_cast<int>(point_value * 10)],
					b_tab_[static_cast<int>(point_value * 10)],
					255
				};
			}
		}
	}
}

auto CoreUtils::Window::setActive(const bool active) const -> void
{
	window_->setActive(active);
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
	return false;
}

auto CoreUtils::Window::display() -> void
{
	if (window_->isOpen()) { window_->display(); }
}

auto CoreUtils::Window::reloadWindow() -> void
{
	needs_reload_ = true;
}

auto CoreUtils::Window::updateInterface() -> void
{
	update_interface_ = true;
}

auto CoreUtils::Window::isOpen() -> bool
{
	return window_->isOpen();
}

auto CoreUtils::Window::toggleControls() -> void
{
	gui_->toggleShowControls();
}
