#pragma once
#include <memory>
#include <Windows.h>

#include "IWindow.h"
#include "Entity.h"
#include <SFML/Graphics/Text.hpp>

namespace CoreUtils
{
	class Engine
	{
	private:
		static auto getExePath() -> std::string;
		Entity::EntityContainer swarm_;
		std::unique_ptr<IWindow> window_;
		std::unique_ptr<sf::Font> system_font_;
	public:
		auto run() -> void;
		Engine();
		~Engine();
	};
}
