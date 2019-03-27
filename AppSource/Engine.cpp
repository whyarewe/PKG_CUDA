#include "Engine.h"
#include "Window.h"
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include "CUDALaplacePropagation.h"
#include <iostream>

CoreUtils::Engine::Engine()
{
	window_ = std::make_unique<Window>(WindowStyles::NonResizable);
}

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

	auto construct_image_from_vector(const std::vector<float>& vec, const uint32_t x_axis_bound,
		const uint32_t y_axis_bound) -> sf::Image
	{
		sf::Image board;
		board.create(x_axis_bound, y_axis_bound, sf::Color::Black);
		for (uint32_t i = 1; i < y_axis_bound - 1; ++i)
		{
			for (uint32_t j = 1; j < x_axis_bound - 1; j++)
			{
				auto pixel_color = sf::Color(
					static_cast<uint8_t>(R(vec[i * x_axis_bound + j])),
					static_cast<uint8_t>(G(vec[i * x_axis_bound + j])),
					static_cast<uint8_t>(B(vec[i * x_axis_bound + j])));

				if (vec[i * x_axis_bound + j] > 125.f)
				{
					pixel_color = sf::Color::White;
				}

				board.setPixel(j, i, pixel_color);
			}
		}
		return board;
	}
}

auto CoreUtils::Engine::run() -> void
{
	uint32_t entity_radius = 1;

	std::vector<float> model;
	model.reserve(window_->getWidth() * window_->getHeight());

	auto font_path = getExePath();
	font_path.resize(font_path.size() - 12);
	font_path += "FontFile.ttf";
	sf::Font font;
	if (!font.loadFromFile(font_path))
	{
		std::exit(0);
	}
	
	sf::Text heaters;
	sf::Text radius;

	heaters.setFont(font);
	radius.setFont(font);

	radius.setCharacterSize(20);
	heaters.setCharacterSize(20);

	radius.setFillColor(sf::Color::White);
	heaters.setFillColor(sf::Color::White);

	radius.move(static_cast<float>(window_->getWidth()) - 180, 30.f);
	heaters.move(static_cast<float>(window_->getWidth() - 180), 10.f);

	auto heaters_count("Heater Count  : " + std::to_string(swarm_.size()));
	auto heater_radius("Heater Radius : " + std::to_string(entity_radius));

	heaters.setString(heaters_count.c_str());
	radius.setString(heater_radius.c_str());

	while (window_->isOpen())
	{
		sf::Event event{};
		while (window_->pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window_->close();
			}
			else if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Left)
				{
					auto x = window_->getMousePosition().x;
					auto y = window_->getMousePosition().y;

					std::cout << "X : " << x << " Y: " << y << std::endl;

					if (x > 1 && x < window_->getWidth() && y > 1 && y < window_->getHeight())
					{
						swarm_.push_back(Entity(x, y, entity_radius));
						auto heaters_count_str("Heater Count  : " + std::to_string(swarm_.size()));
						heaters.setString(heaters_count_str.c_str());
					}
				}
			}
			else if (event.type == sf::Event::EventType::KeyPressed)
			{
				if (event.key.code == sf::Keyboard::Up)
				{
					entity_radius = entity_radius > 20 ? entity_radius : entity_radius += 2;
					auto heater_radius_str("Heater Radius : " + std::to_string(entity_radius));
					radius.setString(heater_radius_str.c_str());
				}
				else if (event.key.code == sf::Keyboard::Down)
				{
					entity_radius = entity_radius == 1 ? entity_radius : entity_radius -= 2;
					auto heater_radius_str("Heater Radius : " + std::to_string(entity_radius));
					radius.setString(heater_radius_str.c_str());
				}
				else if (event.key.code == sf::Keyboard::BackSpace)
				{
					if(!swarm_.empty())
					{
						swarm_.erase(swarm_.begin());
						auto heaters_count_str("Heater Count  : " + std::to_string(swarm_.size()));
						heaters.setString(heaters_count_str.c_str());
					}					
				} 
				else if(event.key.code == sf::Keyboard::F11)
				{
					if(WindowStyles::Resizable == window_->getStyle() || WindowStyles::NonResizable == window_->getStyle())
					{
						window_->setStyle(WindowStyles::FullScreen);
					} 
					else
					{
						window_->setStyle(WindowStyles::NonResizable);
					}
				}
			}
		}

		sf::Image board;
		board.create(window_->getWidth(), window_->getHeight(), sf::Color::Black);

		sf::Texture texture;
		texture.loadFromImage(board);

		sf::Sprite sprite;
		sprite.setTexture(texture, true);

		auto mouse_position = window_->getMousePosition();

		if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
		{
			auto x = mouse_position.x;
			auto y = mouse_position.y;

			if (x > 1 && x < window_->getWidth() && y > 1 && y < window_->getHeight())
			{
				auto left_border = x - entity_radius;
				auto right_border = x + entity_radius;
				auto top_border = y - entity_radius;
				auto bottom_border = y + entity_radius;

				left_border = left_border <= 1 ? 1 : left_border;
				right_border = right_border >= window_->getWidth() ? right_border - 1 : right_border;
				top_border = top_border <= 1 ? 1 : top_border;
				bottom_border = bottom_border >= window_->getHeight() ? window_->getHeight() - 1 : bottom_border;

				for (auto i = top_border; i < bottom_border; ++i)
				{
					for (auto j = left_border; j < right_border; ++j)
					{
						model[i * window_->getWidth() + j] = 255.f;
					}
				}
			}
		}

		for (const auto& entity : swarm_)
		{
			auto left_border = entity.getCoordinates().getX() - entity.getRadius();
			auto right_border = entity.getCoordinates().getX() + entity.getRadius();
			auto top_border = entity.getCoordinates().getY() - entity.getRadius();
			auto bottom_border = entity.getCoordinates().getY() + entity.getRadius();

			left_border = left_border <= 1 ? 1 : left_border;
			right_border = right_border >= window_->getWidth() ? right_border - 1 : right_border;
			top_border = top_border <= 1 ? 1 : top_border;
			bottom_border = bottom_border >= window_->getHeight() ? window_->getHeight() - 1 : bottom_border;

			for (auto i = top_border; i < bottom_border; ++i)
			{
				for (auto j = left_border; j < right_border; ++j)
				{
					model[i * window_->getWidth() + j] = 255.f;
				}
			}
		}
		CUDAHelpers::ComputingData current_board_context{ model,  window_->getWidth() , window_->getHeight(), swarm_ };
		CUDAHelpers::CUDAPropagation::laplace(current_board_context, CUDAHelpers::CUDAPropagation::Device::CPU);
		board = construct_image_from_vector(model, window_->getWidth(), window_->getHeight());
		texture.loadFromImage(board);
		sprite.setTexture(texture, true);

		window_->clear();
		window_->draw(&sprite);
		window_->draw(&radius);
		window_->draw(&heaters);
		window_->display();
	}
}

auto CoreUtils::Engine::getExePath() -> std::string
{
	char result[MAX_PATH];
	return std::string(result, GetModuleFileName(nullptr, result, MAX_PATH));
}

CoreUtils::Engine::~Engine() = default;
