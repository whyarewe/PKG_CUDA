#include "Window.h"
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>

namespace
{
	double R(float x)
	{
		return 255.0605 + (0.02909945 - 255.0605) / std::pow((1 + std::pow((2 * x / 68872.05), 2.133224)), 13205500);
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

	gui_ = std::make_unique<GUI>(*window_, font);
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

auto CoreUtils::Window::generateView(const CUDAHelpers::ComputingData& data) -> void
{
	if (running_view_) { return; }

	if (view_.joinable())
	{
		view_.join();
	}
	view_ = std::thread([&]()
	{
		running_view_ = true;

		sf::Image background_image;
		background_image.create(data.x_axis_bound, data.y_axis_bound, sf::Color::Black);

		while (isOpen() && !needs_reload_)
		{
			constructImageFromVector(background_image, data);
			sf::Texture background_texture;
			background_texture.loadFromImage(background_image);
			sf::Sprite background;
			background.setTexture(background_texture, true);

			clear();
			draw(&background);

			if (update_interface_)
			{
				gui_->setRadius(data.entity_radius);
				gui_->setHeatersCount(data.swarm.size());
				gui_->update();

				update_interface_ = false;
			}

			gui_->display(*window_);
			display();
		}
		running_view_ = false;
	});
}

auto CoreUtils::Window::constructImageFromVector(sf::Image& background_image,
                                                 const CUDAHelpers::ComputingData& data) const -> sf::Image
{
	for (auto i = 1; i < data.y_axis_bound - 1; ++i)
	{
		for (auto j = 1; j < data.x_axis_bound - 1; j++)
		{
			auto pixel_color = sf::Color(
				static_cast<uint8_t>(R(data.board[i * data.x_axis_bound + j])),
				static_cast<uint8_t>(G(data.board[i * data.x_axis_bound + j])),
				static_cast<uint8_t>(B(data.board[i * data.x_axis_bound + j])));

			if (data.board[i * data.x_axis_bound + j] > 125.f)
			{
				pixel_color = sf::Color::White;
			}

			background_image.setPixel(j, i, pixel_color);
		}
	}
	return background_image;
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