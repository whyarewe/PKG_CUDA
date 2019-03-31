#include <iostream>
#include <SFML/Graphics/Text.hpp>

#include "Engine.h"
#include "Window.h"
#include "CUDALaplacePropagation.h"
#include "EntityManager.h"
#include "LevelManager.h"
#include "EventHandler.h"

CoreUtils::Engine::Engine() :
	system_font_(std::make_unique<sf::Font>()),
	event_handler_(std::make_unique<EventHandler>()),
	entity_manager_(std::make_unique<EntityManager>())
{
	auto font_path = getExePath();
	font_path.resize(font_path.size() - 12);
	font_path += "FontFile.ttf";
	if (!system_font_->loadFromFile(font_path))
	{
		std::exit(0);
	}
	window_ = std::make_unique<Window>(WindowStyles::NonResizable, *system_font_);
	level_manager_ = std::make_unique<LevelManager>(window_->getWidth(), window_->getHeight());
}

auto CoreUtils::Engine::run() const -> void
{
	window_->setActive(false);
	while (window_->isOpen())
	{
		event_handler_->intercept(*window_, *entity_manager_, *level_manager_);
		level_manager_->update(*entity_manager_);

		CUDAHelpers::ComputingData board_context{
			level_manager_->getLevel(),
			level_manager_->getXAxisLength(),
			level_manager_->getYAxisLength(),
			entity_manager_->getCurrentRadius(),
			entity_manager_->getAll()
		};

		CUDAHelpers::CUDAPropagation::laplace(board_context, CUDAHelpers::CUDAPropagation::Device::CPU);

		window_->generateView(board_context);
	}
}

auto CoreUtils::Engine::getExePath() -> std::string
{
	char result[MAX_PATH];
	return std::string(result, GetModuleFileName(nullptr, result, MAX_PATH));
}

CoreUtils::Engine::~Engine() = default;
