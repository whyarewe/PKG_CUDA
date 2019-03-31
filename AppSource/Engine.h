#pragma once
#include <memory>
#include <Windows.h>

#include "IWindow.h"
#include <SFML/Graphics/Text.hpp>
#include "IEntityManager.h"
#include "ILevelManager.h"
#include "IEventHandler.h"

namespace CoreUtils
{
	class Engine
	{
	private:
		static auto getExePath() -> std::string;
		std::unique_ptr<IWindow> window_;
		std::unique_ptr<sf::Font> system_font_;
		std::unique_ptr<IEventHandler> event_handler_;
		std::unique_ptr<ILevelManager> level_manager_;
		std::unique_ptr<IEntityManager> entity_manager_;

	public:
		auto run() const -> void;
		Engine();
		~Engine();
	};
}
