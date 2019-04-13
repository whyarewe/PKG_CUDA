#pragma once
#include <memory>
#include <Windows.h>

#include <SFML/Graphics/Text.hpp>

#include "IWindow.h"
#include "IEntityManager.h"
#include "ILevelManager.h"
#include "IEventHandler.h"

namespace CoreUtils
{
	class Engine
	{
	public:
		Engine();
		~Engine();
		auto run() -> void;
		auto reload() -> void;
		static auto getExePath()->std::string;
		
		std::unique_ptr<IWindow> window_;
		std::unique_ptr<sf::Font> system_font_;
		std::unique_ptr<IEventHandler> event_handler_;
		std::unique_ptr<ILevelManager> level_manager_;
		std::unique_ptr<IEntityManager> entity_manager_;
	};
}
