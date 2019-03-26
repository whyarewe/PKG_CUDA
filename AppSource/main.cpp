#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <Windows.h>

#include "CUDASystemInformation.h"
#include "CUDALaplacePropagation.h"
#include "Entity.h"

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

	std::string getExePath()
	{
		char result[MAX_PATH];
		return std::string(result, GetModuleFileName(nullptr, result, MAX_PATH));
	}
}

int main()
{
	CUDAHelpers::CUDASystemInformation system_information;
	system_information.displaySystemDevicesProperites();
	uint32_t x_axis_bound = 600;
	uint32_t y_axis_bound = 600;

	sf::ContextSettings setting;
	setting.antialiasingLevel = 8;
	sf::RenderWindow main_window(sf::VideoMode(x_axis_bound, y_axis_bound), "PKG_CUDA",
	                             sf::Style::Titlebar | sf::Style::Close);
	main_window.setFramerateLimit(60);

	Entity::EntityContainer swarm;
	uint32_t entity_radius = 1;

	std::vector<float> model;
	model.reserve(x_axis_bound * y_axis_bound);

	sf::Image board;
	board.create(x_axis_bound, y_axis_bound, sf::Color::Black);

	sf::Texture texture;
	texture.loadFromImage(board);

	sf::Sprite sprite;
	sprite.setTexture(texture, true);
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

	radius.move(x_axis_bound - 180, 30);
	heaters.move(x_axis_bound - 180, 10);

	auto heaters_count("Heater Count  : " + std::to_string(swarm.size()));
	auto heater_radius("Heater Radius : " + std::to_string(entity_radius));

	heaters.setString(heaters_count.c_str());
	radius.setString(heater_radius.c_str());

	while (main_window.isOpen())
	{
		sf::Event event{};

		while (main_window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				main_window.close();
			}
			else if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Left)
				{
					auto mouse_position = sf::Mouse::getPosition(main_window);

					if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
					{
						uint32_t x = mouse_position.x;
						uint32_t y = mouse_position.y;

						if (x > 1 && x < x_axis_bound && y > 1 && y < y_axis_bound)
						{
							swarm.push_back(Entity(x, y, entity_radius));
							auto heaters_count_str("Heater Count  : " + std::to_string(swarm.size()));
							heaters.setString(heaters_count_str.c_str());
						}
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
					swarm.erase(swarm.begin());
					auto heaters_count_str("Heater Count  : " + std::to_string(swarm.size()));
					heaters.setString(heaters_count_str.c_str());
				}
			}
		}

		auto mouse_position = sf::Mouse::getPosition(main_window);

		if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
		{
			auto x = mouse_position.x;
			auto y = mouse_position.y;

			if (x > 1 && x < x_axis_bound && y > 1 && y < y_axis_bound)
			{
				auto left_border = x - entity_radius;
				auto right_border = x + entity_radius;
				auto top_border = y - entity_radius;
				auto bottom_border = y + entity_radius;

				left_border = left_border <= 1 ? 1 : left_border;
				right_border = right_border >= x_axis_bound ? right_border - 1 : right_border;
				top_border = top_border <= 1 ? 1 : top_border;
				bottom_border = bottom_border >= y_axis_bound ? y_axis_bound - 1 : bottom_border;

				for (auto i = top_border; i < bottom_border; ++i)
				{
					for (auto j = left_border; j < right_border; ++j)
					{
						model[i * x_axis_bound + j] = 255.f;
					}
				}
			}
		}

		for (const auto& entity : swarm)
		{
			auto left_border = entity.getCoordinates().getX() - entity.getRadius();
			auto right_border = entity.getCoordinates().getX() + entity.getRadius();
			auto top_border = entity.getCoordinates().getY() - entity.getRadius();
			auto bottom_border = entity.getCoordinates().getY() + entity.getRadius();

			left_border = left_border <= 1 ? 1 : left_border;
			right_border = right_border >= x_axis_bound ? right_border - 1 : right_border;
			top_border = top_border <= 1 ? 1 : top_border;
			bottom_border = bottom_border >= y_axis_bound ? y_axis_bound - 1 : bottom_border;

			for (auto i = top_border; i < bottom_border; ++i)
			{
				for (auto j = left_border; j < right_border; ++j)
				{
					model[i * x_axis_bound + j] = 255.f;
				}
			}
		}
		CUDAHelpers::ComputingData current_board_context{model, x_axis_bound, y_axis_bound, swarm};
		CUDAHelpers::CUDAPropagation::laplace(current_board_context, CUDAHelpers::CUDAPropagation::Device::CPU);
		board = construct_image_from_vector(model, x_axis_bound, y_axis_bound);
		texture.loadFromImage(board);
		sprite.setTexture(texture, true);

		main_window.clear();
		main_window.draw(sprite);
		main_window.draw(radius);
		main_window.draw(heaters);
		main_window.display();
	}

	return 0;
}
