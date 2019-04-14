#include <iostream>

#include <SFML/Graphics/Text.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "Engine.h"
#include "Window.h"
#include "EntityManager.h"
#include "LevelManager.h"
#include "EventHandler.h"
#include "CUDAPropagation.h"
#include "CUDAUtils.h"


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
	level_manager_ = std::make_unique<LevelManager>(Config::StandardResolution::width, Config::StandardResolution::height);
	window_ = std::make_unique<Window>(WindowStyles::NonResizable, *system_font_);
	propagation_ = std::make_unique<CUDAHelpers::CUDAPropagation>(level_manager_->getXAxisLength(), level_manager_->getYAxisLength());
}

auto CoreUtils::Engine::run() -> void
{
	window_->setActive(false);
	std::vector<double> times;
	auto debug = false;

	while (window_->isOpen())
	{
		event_handler_->intercept(*this, &debug);
		level_manager_->update(*entity_manager_);

		CUDAHelpers::ComputingData board_context{
			level_manager_->getLevel(),
			level_manager_->getXAxisLength(),
			level_manager_->getYAxisLength(),
			entity_manager_->getCurrentRadius(),
			entity_manager_->getAll()
		};

		auto start = std::chrono::system_clock::now();

		propagation_->propagate(board_context,
			Config::Engine_Config::device, Config::Engine_Config::method);

		auto stop = std::chrono::system_clock::now();

		std::chrono::duration<double> time_elapsed = stop - start;
		times.push_back(time_elapsed.count());

		if (debug)
		{
			debug = false;
			std::cout << times.back() << std::endl;
		}

		if (times.size() > 1000)
		{
			times.clear();
		}

		window_->generateView(*level_manager_, *entity_manager_);
	}
}

auto CoreUtils::Engine::reload() -> void
{
	level_manager_.reset(new LevelManager(window_->getWidth(), window_->getHeight()));
	entity_manager_.reset(new EntityManager());

	propagation_.reset();
	propagation_ = std::make_unique<CUDAHelpers::CUDAPropagation>(level_manager_->getXAxisLength(), level_manager_->getYAxisLength());

	run();
}

auto CoreUtils::Engine::getExePath() -> std::string
{
	char result[MAX_PATH];
	return std::string(result, GetModuleFileName(nullptr, result, MAX_PATH));
}

CoreUtils::Engine::~Engine() = default;
