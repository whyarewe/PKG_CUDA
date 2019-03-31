#include "Engine.h"
#include "Window.h"
#include <SFML/Graphics/Text.hpp>
#include "CUDALaplacePropagation.h"
#include <iostream>

CoreUtils::Engine::Engine() :
	system_font_(std::make_unique<sf::Font>())
{
	auto font_path = getExePath();
	font_path.resize(font_path.size() - 12);
	font_path += "FontFile.ttf";
	if (!system_font_->loadFromFile(font_path))
	{
		std::exit(0);
	}
	window_ = std::make_unique<Window>(WindowStyles::NonResizable, *system_font_);
}

auto CoreUtils::Engine::run() -> void
{
	uint32_t entity_radius = 1;
	bool show_controls = false;

	std::vector<float> model;
	model.reserve(window_->getWidth() * window_->getHeight());

	window_->setActive(false);
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
					const auto mouse_position = window_->getMousePosition();

					if (window_->isWithinWindow(mouse_position))
					{
						swarm_.push_back(Entity(mouse_position.x, mouse_position.y, entity_radius, window_->getWidth(),
						                        window_->getHeight()));

						window_->updateInterface();
					}
				}
			}
			else if (event.type == sf::Event::EventType::KeyPressed)
			{
				switch (event.key.code)
				{
				case sf::Keyboard::Up:
					entity_radius = entity_radius > 20 ? entity_radius : entity_radius += 2;
					window_->updateInterface();
					break;
				case sf::Keyboard::Down:
					entity_radius = entity_radius == 1 ? entity_radius : entity_radius -= 2;
					window_->updateInterface();
					break;
				case sf::Keyboard::BackSpace:
					if (!swarm_.empty())
					{
						swarm_.erase(swarm_.begin());
						window_->updateInterface();
					}
					break;
				case sf::Keyboard::F11:
					switch (window_->getStyle())
					{
					case WindowStyles::Resizable:
					case WindowStyles::NonResizable:
						window_->setStyle(WindowStyles::FullScreen);
						window_->reloadWindow();
						break;
					case WindowStyles::FullScreen:
						window_->setStyle(WindowStyles::NonResizable);
						window_->reloadWindow();
						break;
					}
					break;
				case sf::Keyboard::I:
					window_->toggleControls();
					break;
				default:
					break;
				}
			}
		}
		if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
		{
			const auto mouse_position = window_->getMousePosition();
			if (window_->isWithinWindow(mouse_position))
			{
				Entity volatile_entity(mouse_position.x, mouse_position.y, entity_radius, window_->getWidth(),
				                       window_->getHeight());

				for (auto i = volatile_entity.getDimensions().getTopBorder();
				     i < volatile_entity.getDimensions().getBottomBorder(); ++i)
				{
					for (auto j = volatile_entity.getDimensions().getLeftBorder();
					     j < volatile_entity.getDimensions().getRightBorder(); ++j)
					{
						model[i * window_->getWidth() + j] = 255.f;
					}
				}
			}
		}

		for (const auto& entity : swarm_)
		{
			for (auto i = entity.getDimensions().getTopBorder(); i < entity.getDimensions().getBottomBorder(); ++i)
			{
				for (auto j = entity.getDimensions().getLeftBorder(); j < entity.getDimensions().getRightBorder(); ++j)
				{
					model[i * window_->getWidth() + j] = 255.f;
				}
			}
		}

		CUDAHelpers::ComputingData board_context{
			model,
			window_->getWidth(),
			window_->getHeight(),
			entity_radius,
			swarm_
		};

		CUDAHelpers::CUDAPropagation::laplace(board_context, CUDAHelpers::CUDAPropagation::Device::GPU);

		window_->generateView(board_context);
	}
}

auto CoreUtils::Engine::getExePath() -> std::string
{
	char result[MAX_PATH];
	return std::string(result, GetModuleFileName(nullptr, result, MAX_PATH));
}

CoreUtils::Engine::~Engine() = default;
