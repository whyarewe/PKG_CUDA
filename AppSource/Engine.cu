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
}

auto CoreUtils::Engine::run() const -> void
{
	window_->setActive(false);
	std::vector<double> times;
	auto output_time = false;

	float* data = nullptr;
	float* out_data = nullptr;
	float* host_data = nullptr;

	cudaMalloc((void**)&data, level_manager_->getXAxisLength() * level_manager_->getYAxisLength() * sizeof(float));
	cudaMalloc((void**)&out_data, level_manager_->getXAxisLength() * level_manager_->getYAxisLength() * sizeof(float));

	while (window_->isOpen())
	{
		event_handler_->intercept(*window_, *entity_manager_, *level_manager_, &output_time);
		level_manager_->update(*entity_manager_);

		CUDAHelpers::ComputingData board_context{
			level_manager_->getLevel(),
			level_manager_->getXAxisLength(),
			level_manager_->getYAxisLength(),
			entity_manager_->getCurrentRadius(),
			entity_manager_->getAll()
		};

		auto start = std::chrono::system_clock::now();

		CUDAHelpers::CUDAPropagation::propagate(data, out_data, board_context,
			CUDAHelpers::CUDAPropagation::Device::GPU, CUDAHelpers::CUDAPropagation::Method::FIS);

		auto stop = std::chrono::system_clock::now();

		std::chrono::duration<double> time_elapsed = stop - start;
		times.push_back(time_elapsed.count());

		if (output_time)
		{
			output_time = false;
			std::cout << times.back() << std::endl;
		}

		if (times.size() > 1000)
		{
			times.clear();
		}

		window_->generateView(*level_manager_, *entity_manager_);
	}

	cudaFree(data);
	cudaFree(out_data);
	cudaFree(host_data);
}

auto CoreUtils::Engine::getExePath() -> std::string
{
	char result[MAX_PATH];
	return std::string(result, GetModuleFileName(nullptr, result, MAX_PATH));
}

CoreUtils::Engine::~Engine() = default;
